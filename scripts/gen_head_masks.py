"""在【原始】LeRobot 数据目录上逐帧生成头部相机 mask, 直接用部署同一个 airbot/hf_mask.HFMaskPredictor,
保证训练 mask 与真机 runner 的 mask 是同一段代码。输出布局与 hqi 脚本 / scripts/map_masks_to_packed.py 一致:
    <out_root>/<源目录名>/videos/chunk-000/<video-key>/episode_XXXXXX/all_frames/<槽位>/frame_XXXXXX.png
已存在的帧跳过, 中断后重跑即可续上。只写 png(训练 loader 读 png)。

用法(conda LLM 环境, 显卡空闲时)。
hqi 倒水(打包后的 hqi_pour50, 俯视相机 cam_high):
    cd ~/openpi_hqi_pour && HF_HUB_OFFLINE=1 python scripts/gen_head_masks.py --dataset-root ~/pi_data/hqi_pour50 \
        --out-root ~/pi_data/hqi_pour50_masks --video-key cam_high --box-threshold 0.2 \
        --class "pitcher=orange juice pitcher" --class "cup=small glass cup" \
        --size-slot pitcher=0.025:0.15 --size-slot cup=0:0.025
擦黑板:
    cd ~/openpi && HF_HUB_OFFLINE=1 python scripts/gen_head_masks.py \
        --dataset-root ~/pi_data/parallel/clean_board_task/airbot_wipe_varied \
        --out-root ~/pi_data/parallel/clean_board_task_masks_grounded_sam \
        --class "eraser=black eraser|eraser" --class "whiteboard=white board|whiteboard" --inside eraser=whiteboard \
        --inside-margin 0.3
(老 50 条 airbot_wipe_data / airbot_wipe_data_20260616_171940 生成时是 --inside-margin 0。抽查 400 帧,
 改成 0.3 只有 1 帧不同(夹持中夹爪上部被当板擦, 框中心在白板上沿外 2px); 部署在新机位, 老的不重跑)
"""
import argparse
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "airbot"))
from hf_mask import HFMaskPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", action="append", required=True, type=pathlib.Path, help="原始子数据集目录, 可重复")
    ap.add_argument("--out-root", required=True, type=pathlib.Path)
    ap.add_argument("--class", dest="classes", action="append", required=True, help="槽位名=检测词1|检测词2, 顺序即槽位顺序")
    ap.add_argument("--inside", action="append", default=[], help="槽位=外框槽位: 前者框中心须在后者框内")
    ap.add_argument("--inside-margin", type=float, default=0.0, help="外框每边外扩比例(按外框宽/高), 擦黑板用 0.3")
    ap.add_argument("--multi", action="append", default=[], help="该槽位取全部实例的 mask 并集(挂杯: mug), 可重复")
    ap.add_argument("--max-instances", type=int, default=4)
    ap.add_argument("--max-box-frac", type=float, default=1.0, help="丢掉面积超过整图该比例的框(挂杯 0.6), 1.0 = 不过滤")
    ap.add_argument("--subtract", action="append", default=[], help="槽位=槽位2: 前者 mask 扣掉后者(挂杯 mug_tree=mug), 可重复")
    ap.add_argument("--size-slot", action="append", default=[], help="槽位=最小面积比:最大面积比, 给了就按框大小分槽位(hqi 倒水), 须覆盖全部槽位")
    ap.add_argument("--box-threshold", type=float, default=0.28)
    ap.add_argument("--video-key", default="image")
    ap.add_argument("--max-episodes", type=int, default=None, help="调试用")
    ap.add_argument("--max-frames", type=int, default=None, help="调试用")
    a = ap.parse_args()

    prompts = {}
    for c in a.classes:
        k, v = c.split("=", 1)
        prompts[k.strip()] = [p.strip() for p in v.split("|") if p.strip()]
    inside = dict(kv.split("=", 1) for kv in a.inside)
    subtract = {}
    for kv in a.subtract:
        k, v = kv.split("=", 1); subtract.setdefault(k.strip(), []).append(v.strip())
    size_slots = {}
    for kv in a.size_slot:
        k, v = kv.split("=", 1); lo, hi = v.split(":"); size_slots[k.strip()] = (float(lo), float(hi))
    masker = HFMaskPredictor(prompts, inside=inside, inside_margin=a.inside_margin, box_threshold=a.box_threshold,
                             multi=a.multi, max_instances=a.max_instances, max_box_frac=a.max_box_frac, subtract=subtract,
                             size_slots=size_slots or None)
    print(f"槽位 {prompts}  约束 {inside} 外扩 {a.inside_margin}  多实例 {a.multi}(≤{a.max_instances})  大框过滤 {a.max_box_frac}"
          f"  扣除 {subtract}  按大小分槽 {size_slots}  阈值 {a.box_threshold}", flush=True)

    for root in a.dataset_root:
        root = root.expanduser()
        vids = sorted((root / "videos" / "chunk-000" / a.video_key).glob("episode_*.mp4"))[: a.max_episodes]
        if not vids:
            sys.exit(f"没找到视频: {root}/videos/chunk-000/{a.video_key}/episode_*.mp4")
        out_sub = a.out_root.expanduser() / root.name / "videos" / "chunk-000" / a.video_key
        summary = {"prompts": prompts, "inside": inside, "inside_margin": a.inside_margin, "multi": a.multi,
                   "max_instances": a.max_instances, "max_box_frac": a.max_box_frac, "subtract": subtract,
                   "size_slots": size_slots, "box_threshold": a.box_threshold, "episodes": {}}
        for vi, vp in enumerate(vids):
            ep_dir = out_sub / vp.stem / "all_frames"
            for k in prompts:
                (ep_dir / k).mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(vp))
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            hits, n, t0 = np.zeros(len(prompts), int), 0, time.time()
            while True:
                ok, bgr = cap.read()
                if not ok or (a.max_frames is not None and n >= a.max_frames):
                    break
                pngs = [ep_dir / k / f"frame_{n:06d}.png" for k in prompts]
                if all(p.exists() for p in pngs):
                    hits += [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).any() for p in pngs]
                else:
                    m = masker(bgr[:, :, ::-1].copy())
                    for p, mk in zip(pngs, m):
                        cv2.imwrite(str(p), (mk > 0).astype(np.uint8) * 255)
                    hits += (m.reshape(len(prompts), -1) > 0).any(1)
                n += 1
            cap.release()
            if a.max_frames is None and n != n_total:
                print(f"  ⚠ {vp.name}: 读到 {n} 帧, 元数据 {n_total} 帧", flush=True)
            summary["episodes"][vp.stem] = {"frames": n, **{f"{k}_detected": int(h) for k, h in zip(prompts, hits)}}
            print(f"[{root.name} {vi + 1}/{len(vids)}] {vp.stem}: {n} 帧, 检出 "
                  + ", ".join(f"{k} {h}/{n}" for k, h in zip(prompts, hits)) + f"  ({time.time() - t0:.0f}s)", flush=True)
        (out_sub.parents[2] / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("完成", flush=True)


if __name__ == "__main__":
    main()
