"""部署用 mask 生成器, 与训练 mask 完全同源: 逻辑和默认参数逐行照搬 hqi 的
tools/generate_grounded_sam_masks_brush.py(HF IDEA-Research/grounding-dino-base + facebook/sam-vit-base,
box 0.28 / text 0.24, 推理前把长边缩到 512, 每个类取得分最高的框, SAM 取 iou 最高的 mask, 检不到 = 全零)。

    masker = HFMaskPredictor({"blue_dustpan": ["blue dustpan", "dustpan"], "hand_brush": ["hand brush", "brush"]})
    masks = masker(rgb)      # (H,W,3) uint8 RGB -> (slots,H,W) float32 0/1, 槽位顺序 = 字典顺序

inside={"eraser": "whiteboard"}(可选, 默认不用, 扫地不受影响): eraser 的框中心必须落在 whiteboard 的框内, 否则当作没检到。
擦黑板时板擦被夹爪挡住, "black eraser" 会去框后面架子上的黑色物体; 训练 mask 和部署必须用同一套设置。
inside_margin: 外框每边外扩的比例(按外框宽/高)。板擦放在白板旁边桌上时框中心在白板框外最远 0.17,
误检(新机位左上角黑物 / 老机位后面架子)最近 0.43 / 0.55, 取 0.3。默认 0 = 严格在框内。
multi={"mug"}(可选, 默认空): 这些槽位取全部实例(框按 IoU>0.5 去重, 最多 max_instances 个)的 mask 并集, 而不是只取最高分一个。
挂杯有两个一样的杯子, 只取最高分会在两个杯子间跳。
max_box_frac(可选, 默认 1.0 = 不过滤): 面积超过整图该比例的框直接丢掉(挂杯 "black stand" 偶尔给出整图框)。
subtract={"mug_tree": ["mug"]}(可选, 默认空): 前者的 mask 扣掉后者(SAM 从杯架框分割时会把挂上去的杯子也算进杯架)。
size_slots={"pitcher": (0.025, 0.15), "cup": (0.0, 0.025)}(可选, 默认空 = 上面的原逻辑): 按框面积占整图的比例分槽位,
见 _boxes_by_size。hqi 倒水数据(俯视 cam_high)用它, 配 box_threshold=0.2。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class HFMaskPredictor:
    def __init__(self, class_prompts: dict[str, list[str]], device: str = "cuda",
                 dino_model_id: str = "IDEA-Research/grounding-dino-base", sam_model_id: str = "facebook/sam-vit-base",
                 box_threshold: float = 0.28, text_threshold: float = 0.24, inference_max_side: int = 512,
                 inside: dict[str, str] | None = None, inside_margin: float = 0.0,
                 multi: tuple[str, ...] | list[str] | set[str] = (), max_instances: int = 4, max_box_frac: float = 1.0,
                 subtract: dict[str, list[str]] | None = None,
                 size_slots: dict[str, tuple[float, float]] | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.class_prompts = {k: [p.strip() for p in v if p.strip()] for k, v in class_prompts.items()}
        self.inside, self.inside_margin = dict(inside or {}), float(inside_margin)
        for k, v in self.inside.items():
            if k not in self.class_prompts or v not in self.class_prompts:
                raise ValueError(f"inside {k}->{v}: 两个都必须是槽位名 {list(self.class_prompts)}")
        self.multi, self.max_instances, self.max_box_frac = set(multi), int(max_instances), float(max_box_frac)
        if not self.multi <= set(self.class_prompts):
            raise ValueError(f"multi {sorted(self.multi)} 必须是槽位名 {list(self.class_prompts)}")
        if self.multi & (set(self.inside) | set(self.inside.values())):
            raise ValueError("multi 槽位不支持 inside 约束")
        self.subtract = {k: list(v) for k, v in (subtract or {}).items()}
        for k, vs in self.subtract.items():
            if k not in self.class_prompts or not set(vs) <= set(self.class_prompts):
                raise ValueError(f"subtract {k}->{vs}: 都必须是槽位名 {list(self.class_prompts)}")
        self.size_slots = {k: (float(v[0]), float(v[1])) for k, v in (size_slots or {}).items()}
        if self.size_slots:
            if set(self.size_slots) != set(self.class_prompts):
                raise ValueError(f"size_slots {sorted(self.size_slots)} 必须覆盖全部槽位 {list(self.class_prompts)}")
            if self.inside or self.multi or self.subtract or self.max_box_frac < 1.0:
                raise ValueError("size_slots 模式不和 inside/multi/subtract/max_box_frac 混用")
        self.device = torch.device(device)
        self.box_threshold, self.text_threshold, self.max_side = box_threshold, text_threshold, inference_max_side
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        def _load(cls, model_id):
            # 新版 transformers(>=4.56, 训练 mask 用的 5.x)叫 dtype; openpi pyproject 锁的 4.53.2 只认 torch_dtype
            try:
                return cls.from_pretrained(model_id, dtype=dtype)
            except TypeError:
                return cls.from_pretrained(model_id, torch_dtype=dtype)

        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = _load(AutoModelForZeroShotObjectDetection, dino_model_id).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = _load(SamModel, sam_model_id).to(self.device).eval()

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.max_side <= 0 or max(h, w) <= self.max_side:
            return rgb
        s = float(self.max_side) / float(max(h, w))
        return cv2.resize(rgb, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)

    def _boxes(self, image_pil: Image.Image) -> dict[str, np.ndarray | None]:
        phrases: list[str] = []
        for variants in self.class_prompts.values():
            for p in variants:
                p = p.strip().lower()
                if p and p not in phrases:
                    phrases.append(p)
        text = ". ".join(phrases)
        text = text if text.endswith(".") else text + "."
        inputs = self.dino_processor(images=image_pil, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            outputs = self.dino_model(**inputs)
        result = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image_pil.size[::-1]], text_labels=[phrases])[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = [str(x).lower() for x in result.get("text_labels", result.get("labels", []))]
        W, H = image_pil.size
        cand: dict[str, list[tuple[float, np.ndarray]]] = {k: [] for k in self.class_prompts}
        for box, score, label in zip(boxes, scores, labels):
            if self.max_box_frac < 1.0 and (box[2] - box[0]) * (box[3] - box[1]) > self.max_box_frac * W * H:
                continue
            for key, variants in self.class_prompts.items():
                if any(v.lower() in label for v in variants):
                    cand[key].append((float(score), box))
        out: dict = {}
        for k, c in cand.items():
            c.sort(key=lambda x: -x[0])          # 稳定排序: 同分保持原顺序, 与原"严格大于才替换"一致
            if k not in self.multi:
                out[k] = c[0][1] if c else None
                continue
            keep: list[np.ndarray] = []
            for _, b in c:
                if len(keep) >= self.max_instances:
                    break
                if all(_iou(b, q) <= 0.5 for q in keep):
                    keep.append(b)
            out[k] = keep or None
        return out

    def _segment(self, image_pil: Image.Image, box: np.ndarray) -> np.ndarray:
        sam_inputs = self.sam_processor(images=image_pil, input_boxes=[[box.tolist()]], return_tensors="pt")
        sam_inputs = {k: v.to(self.device) for k, v in sam_inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            out = self.sam_model(**sam_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            out.pred_masks.detach().cpu(), sam_inputs["original_sizes"].detach().cpu(),
            sam_inputs["reshaped_input_sizes"].detach().cpu())
        ious = out.iou_scores.detach().cpu().float().numpy()[0][0]
        return masks[0][0].numpy()[int(np.argmax(ious))] > 0

    def _boxes_by_size(self, image_pil: Image.Image) -> dict[str, np.ndarray | None]:
        """size_slots 模式(hqi 倒水数据: 水壶和玻璃杯都是透明容器, DINO 分不清是哪个词):
        合并提示词跑一次 DINO, 每个 query 的分数 = 所有短语 token 概率的最大值(不看它像哪个词),
        再按框面积占整图的比例区间定槽位, 在所有组合里取总分最高的一组: 各槽位框 IoU < 0.3、可以留空。
        分数门槛 = box_threshold。"""
        phrases: list[str] = []
        for variants in self.class_prompts.values():
            for p in variants:
                p = p.strip().lower()
                if p and p not in phrases:
                    phrases.append(p)
        text = ". ".join(phrases) + "."
        inputs = self.dino_processor(images=image_pil, text=text, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        spans = []
        for ph in phrases:
            pid = self.dino_processor.tokenizer(ph, add_special_tokens=False)["input_ids"]
            spans += [(s, s + len(pid)) for s in range(len(ids)) if ids[s:s + len(pid)] == pid][:1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            outputs = self.dino_model(**inputs)
        prob = outputs.logits[0].float().sigmoid().cpu().numpy()                 # (Q, T)
        score = np.max(np.stack([prob[:, a:b].max(1) for a, b in spans], 1), 1)  # (Q,)
        W, H = image_pil.size
        cx, cy, bw, bh = outputs.pred_boxes[0].float().cpu().numpy().T
        xyxy = np.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], 1)
        frac = (bw * bh)
        names = list(self.class_prompts)
        cands = []
        for k in names:
            lo, hi = self.size_slots[k]
            idx = np.flatnonzero((score >= self.box_threshold) & (frac >= lo) & (frac <= hi))
            cands.append([None, *idx[np.argsort(-score[idx])][:10].tolist()])
        best, best_sel = 0.0, [None] * len(names)

        def rec(si, sel, tot):
            nonlocal best, best_sel
            if si == len(names):
                if tot > best:
                    best, best_sel = tot, list(sel)
                return
            for q in cands[si]:
                if q is not None and any(p is not None and _iou(xyxy[q], xyxy[p]) >= 0.3 for p in sel):
                    continue
                rec(si + 1, sel + [q], tot + (score[q] if q is not None else 0.0))

        rec(0, [], 0.0)
        return {k: (xyxy[q] if q is not None else None) for k, q in zip(names, best_sel)}

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        small = self._resize(rgb)
        pil = Image.fromarray(small)
        boxes = self._boxes_by_size(pil) if self.size_slots else self._boxes(pil)
        for k, v in self.inside.items():
            b, outer = boxes[k], boxes[v]
            if b is None:
                continue
            if outer is None:
                boxes[k] = None
                continue
            mx, my = self.inside_margin * (outer[2] - outer[0]), self.inside_margin * (outer[3] - outer[1])
            if not (outer[0] - mx <= (b[0] + b[2]) / 2 <= outer[2] + mx and outer[1] - my <= (b[1] + b[3]) / 2 <= outer[3] + my):
                boxes[k] = None
        out = []
        for key in self.class_prompts:
            if boxes[key] is None:
                m = np.zeros((h, w), bool)
            else:
                bl = boxes[key] if key in self.multi else [boxes[key]]
                m = np.zeros(small.shape[:2], bool)
                for b in bl:
                    m |= self._segment(pil, b)
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            out.append(m.astype(np.float32))
        names = list(self.class_prompts)
        for k, vs in self.subtract.items():           # 例: 杯架 mask 扣掉杯子(SAM 会把挂上去的杯子算进杯架)
            i = names.index(k)
            for v in vs:
                out[i][out[names.index(v)] > 0] = 0.0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return np.stack(out, axis=0)
