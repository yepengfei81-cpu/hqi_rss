#!/usr/bin/env python3
"""hqi 倒水三个模型的关节空间 rollout(与机器人无关的骨架, 机器人部分自己接)。

模型(同一份 50 条数据, 只差 mask 方法):
  pi05_hqi_pour50             baseline
  pi05_hqi_pour50_mask        mask prompt + MCR(hqi 的方法)
  pi05_hqi_pour50_controlvla  ControlVLA(ybao 的实现)
权重: 集群 $P/openpi_hqi_pour/checkpoints/<config>/<exp>/19999, 部署只需要其中的 params/ 和 assets/(train_state/ 不用)。

一、先离线自检(不连机器人): 用 hqi_pour50 的一条示教当"机器人"喂给模型, 每 10 步开环预测一次, 和示教动作比。
    python rollout_hqi_pour.py --config-name pi05_hqi_pour50_mask \
        --checkpoint checkpoints/pi05_hqi_pour50_mask/pour50_mask_v1/19999 \
        --robot replay --replay-root ~/pi_data/hqi_pour50 --replay-episode 3 \
        --replay-mask-root ~/pi_data/hqi_pour50_masks/hqi_pour50
二、真机: 先把下面 RealRobot 的 5 个方法实现好, 然后
    python rollout_hqi_pour.py --config-name ... --checkpoint ... --robot real --dry-run   # 只推理+打印, 不动
    python rollout_hqi_pour.py --config-name ... --checkpoint ... --robot real             # 真动; 终端输入 q 回车 = 结束本局
详细说明见同目录 ROLLOUT_HQI.md。
"""
import os
import sys

# mask / ControlVLA 模型要在同一进程里跑 DINO+SAM(torch): JAX 不预占显存、用完即还, 否则 12G 卡上 DINO 会 OOM
if any(s.endswith(("_mask", "_controlvla")) for s in sys.argv):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import pathlib
import queue
import threading
import time

import cv2
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "airbot"))   # hf_mask.py(训练 mask 和部署 mask 是同一个文件)

CONFIGS = ("pi05_hqi_pour50", "pi05_hqi_pour50_mask", "pi05_hqi_pour50_controlvla")
PROMPT = "pour the orange juice from the pitcher into the glass cup"   # = hqi_pour50/meta/tasks.jsonl, 必须逐字一致
FPS = 10                                                               # 训练数据的采样频率 = 每个动作步 0.1 s
CAM_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAM_HW = {"cam_high": (540, 960), "cam_left_wrist": (480, 848), "cam_right_wrist": (480, 848)}   # 训练图像 (高, 宽)
# 14 维顺序(state 和 action 一样): [左 j1..j6, 左夹爪, 右 j1..j6, 右夹爪], 关节 = 弧度, 夹爪 = 0..1(与录制时同一约定)
JOINT_NAMES = [f"L_j{i}" for i in range(1, 7)] + ["L_grip"] + [f"R_j{i}" for i in range(1, 7)] + ["R_grip"]
GRIP_IDX = (6, 13)
ARM_IDX = tuple(i for i in range(14) if i not in GRIP_IDX)
# 50 条示教第 0 帧 state 的中位数(几乎就是零位 + 两个夹爪张开), --start-pose dataset 用
START_POSE = np.array([-0.0742, 0.0071, 0.0055, -0.0309, 0.0238, 0.0312, 0.9941,
                       0.0429, 0.0158, 0.0109, -0.0575, 0.0038, 0.0521, 0.9993])
# 训练 mask 的生成参数(scripts/gen_head_masks.py 文件头那条命令), 部署必须逐项一致:
# 同一个检测器(HF grounding-dino-base + sam-vit-base), 同样的检测词、按框面积分槽位、阈值 0.2
MASK_PROMPTS = {"pitcher": ["orange juice pitcher"], "cup": ["small glass cup"]}
MASK_SIZE_SLOTS = {"pitcher": (0.025, 0.15), "cup": (0.0, 0.025)}
MASK_BOX_THRESHOLD = 0.2


# ============================== 机器人接口 ==============================
class Robot:
    """runner 只通过这 5 个方法碰硬件。realtime=False(离线回放)时 runner 不 sleep、不插值。"""

    realtime = True

    def read_images(self) -> dict:
        """{'cam_high','cam_left_wrist','cam_right_wrist'} -> (H, W, 3) uint8 **RGB**, 分辨率同训练(见 CAM_HW)。"""
        raise NotImplementedError

    def read_joints(self) -> np.ndarray:
        """从臂(follower)当前位置, (14,) 布局见 JOINT_NAMES。"""
        raise NotImplementedError

    def command_joints(self, q: np.ndarray) -> None:
        """把 (14,) 目标位置发给从臂, 非阻塞。训练动作 = 主臂(leader)位置, 所以这里就是"让模型代替主臂"。"""
        raise NotImplementedError

    def hold(self) -> None:
        """停在当前位置(一局结束 / 急停时调用)。"""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def advance(self) -> None:
        """每执行完一个动作步调用一次(只有回放用得到)。"""

    def done(self) -> bool:
        """回放到头了没有(真机永远 False)。"""
        return False


class RealRobot(Robot):
    """TODO(hqi): 接你们自己的机械臂和相机。要求:
    - 相机: 和采集时同一台相机、同一机位、同分辨率(俯视 960x540, 手腕 848x480), 返回 RGB(OpenCV 读出来是 BGR, 要转)。
    - read_joints: 读 /robot/arm_{left,right}_follow/joint_states 的 position(6 关节 + 夹爪 0..1), 按 JOINT_NAMES 拼 14 维。
      图像和关节要尽量同一时刻: runner 先 read_images() 紧接着 read_joints(), 别在两者之间插耗时操作。
    - command_joints: 最省事的做法是把模型动作当成主臂位置发出去, 走遥操作时 "主臂 -> 从臂" 的同一条控制链
      (训练里 action = 主臂在同一时刻的位置)。runner 会以 --servo-hz 的频率发线性插值后的目标。
    - hold: 让从臂停在当前位置(比如把当前 position 作为目标再发一次)。
    """

    def __init__(self, args):
        raise NotImplementedError("先实现 RealRobot(见类注释), 或者用 --robot replay 离线自检")


class ReplayRobot(Robot):
    """离线自检: 把 hqi_pour50 的一条示教当"机器人"。图像和 state 按示教时间走(不跟随指令), 所以是开环检查。"""

    realtime = False

    def __init__(self, root, episode):
        import pandas as pd

        root = pathlib.Path(root).expanduser()
        df = pd.read_parquet(root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
        self.state = np.stack(df["state"].to_numpy()).astype(np.float32)
        self.actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
        self.caps = {k: cv2.VideoCapture(str(root / "videos" / "chunk-000" / k / f"episode_{episode:06d}.mp4"))
                     for k in CAM_KEYS}
        self.pos = {k: -1 for k in CAM_KEYS}
        self.frame = {}
        self.t = 0
        self.last_cmd = None
        print(f">>> 回放 episode {episode}: {len(self.state)} 帧({len(self.state) / FPS:.1f}s)", flush=True)

    def done(self):
        return self.t >= len(self.state)

    def read_images(self):
        for k, cap in self.caps.items():            # 只会往前读, 不缓存整段视频
            while self.pos[k] < self.t:
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"{k} 视频在第 {self.pos[k] + 1} 帧读完了, parquet 有 {len(self.state)} 帧")
                self.pos[k] += 1
                self.frame[k] = bgr[:, :, ::-1].copy()
        return dict(self.frame)

    def read_joints(self):
        return self.state[self.t].copy()

    def command_joints(self, q):
        self.last_cmd = np.asarray(q)

    def hold(self):
        pass

    def advance(self):
        self.t += 1

    def close(self):
        for cap in self.caps.values():
            cap.release()


# ============================== 模型 / mask ==============================
def load_policy(config_name, checkpoint, prompt):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    parts = pathlib.Path(checkpoint).resolve().parts
    if len(parts) >= 3 and parts[-3] != config_name:   # checkpoints/<config>/<exp>/<step>
        print(f">>> ⚠ checkpoint 路径里的配置名是 '{parts[-3]}', --config-name 是 '{config_name}', 多半拿错了", flush=True)
    # 归一化统计量从 checkpoint/assets 读(与训练同一份), 输入/输出变换由 config 决定(关节转增量、夹爪绝对, 输出还原成绝对)
    policy = _policy_config.create_trained_policy(cfg, pathlib.Path(checkpoint), default_prompt=prompt)
    return cfg, policy


def make_masker(cfg):
    if not (getattr(cfg.model, "use_mask_prompts", False) or getattr(cfg.model, "use_controlvla", False)):
        return None
    names = tuple(cfg.data.mask_object_names)
    if names != tuple(MASK_PROMPTS):
        sys.exit(f"config 的槽位 {names} 与 MASK_PROMPTS {tuple(MASK_PROMPTS)} 不一致")
    from hf_mask import HFMaskPredictor

    print(f">>> 加载 mask 生成器(HF grounding-dino-base + sam-vit-base), 槽位 {names}, "
          f"按框面积分槽 {MASK_SIZE_SLOTS}, 阈值 {MASK_BOX_THRESHOLD}(与训练 mask 同一套参数)", flush=True)
    return HFMaskPredictor(MASK_PROMPTS, box_threshold=MASK_BOX_THRESHOLD, size_slots=MASK_SIZE_SLOTS)


_warned_shape = set()
_jit_done = [False]   # 只有进程里第一次推理要 JIT 编译


def get_observation(robot, masker, prompt):
    imgs = robot.read_images()
    q = np.asarray(robot.read_joints(), np.float32)   # 紧跟取图读状态, 在 DINO+SAM 之前(否则状态比图像新 ~0.3 s)
    if q.shape != (14,):
        raise ValueError(f"read_joints 应返回 (14,), 收到 {q.shape}")
    obs = {"observation/state": q, "prompt": prompt}
    for k in CAM_KEYS:
        im = np.asarray(imgs[k])
        if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
            raise ValueError(f"{k} 应为 (H,W,3) uint8 RGB, 收到 {im.dtype} {im.shape}")
        if im.shape[:2] != CAM_HW[k] and k not in _warned_shape:
            _warned_shape.add(k)
            print(f">>> ⚠ {k} 分辨率 {im.shape[:2]} 与训练 {CAM_HW[k]} 不同(模型会缩放, 但视野/比例要和采集一致)", flush=True)
        obs[f"observation/{k}"] = im
    masks = None
    if masker is not None:
        masks = masker(obs["observation/cam_high"])          # (2, H, W) float32 0/1, 槽位顺序 pitcher, cup
        obs["observation/base_region_mask"] = masks          # 只给俯视相机; 推理时不做 MCR(图像保持原样)
    return obs, masks


# ============================== 执行 ==============================
def ensemble(chunks, macro, m):
    """时序集成: 所有覆盖当前步的 chunk 按新旧加权(新的权重大)。--chunk 10 时同一时刻只有一个 chunk, 等于直接执行。"""
    preds, weights = [], []
    n = len(chunks)
    for idx, (start, cw) in enumerate(chunks):
        off = macro - start
        if 0 <= off < len(cw):
            preds.append(cw[off])
            weights.append(float(np.exp(-m * ((n - 1) - idx))))
    if not preds:
        return None
    w = np.asarray(weights) / np.sum(weights)
    return (w[:, None] * np.asarray(preds)).sum(0)


def safety(target, last, max_step_rad):
    """单步关节位移限幅(防野跳; 示教里右臂单步最大 17°) + 夹爪限到 0..1。"""
    target = np.asarray(target, np.float64).copy()
    d = target[list(ARM_IDX)] - last[list(ARM_IDX)]
    target[list(ARM_IDX)] = last[list(ARM_IDX)] + np.clip(d, -max_step_rad, max_step_rad)
    target[list(GRIP_IDX)] = np.clip(target[list(GRIP_IDX)], 0.0, 1.0)
    clipped = bool(np.any(np.abs(d) > max_step_rad))
    return target, clipped


def servo(robot, q_from, q_to, dt, hz, console):
    """一个动作步(0.1 s / speed_scale)内从 q_from 线性插值到 q_to, 按 hz 下发。hz<=0: 只发一次目标。"""
    if not robot.realtime:
        robot.command_joints(q_to)
        return False
    n = max(1, int(round(hz * dt))) if hz > 0 else 1
    for s in range(n):
        t0 = time.monotonic()
        a = (s + 1) / n
        robot.command_joints((1 - a) * q_from + a * q_to)
        if console.stop_requested():
            return True
        time.sleep(max(0.0, dt / n - (time.monotonic() - t0)))
    return False


def move_to(robot, q_target, duration, hz):
    """开局前慢慢移到 q_target(关节空间线性插值)。"""
    q0 = np.asarray(robot.read_joints(), np.float64)
    n = max(1, int(duration * hz))
    for s in range(n):
        t0 = time.monotonic()
        a = (s + 1) / n
        robot.command_joints((1 - a) * q0 + a * q_target)
        time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))
    time.sleep(0.5)


class ReplayEval:
    """离线自检的统计: 预测 chunk vs 示教动作 actions[t:t+H]; 对照 = 原地不动(一直取 state[t])。"""

    def __init__(self, robot, mask_root, episode):
        self.r = robot
        self.err, self.hold_err, self.grip_err = [], [], []
        self.mask_root = pathlib.Path(mask_root).expanduser() if mask_root else None
        self.episode = episode
        self.iou = {k: [] for k in MASK_PROMPTS}
        self.det_agree = {k: 0 for k in MASK_PROMPTS}
        self.n_mask = 0

    def add(self, t, cw, masks):
        gt = self.r.actions[t:t + len(cw)]
        k = len(gt)
        self.err.append(np.abs(cw[:k] - gt)[:, list(ARM_IDX)])
        self.hold_err.append(np.abs(self.r.state[t][None] - gt)[:, list(ARM_IDX)])
        self.grip_err.append(np.abs(cw[:k] - gt)[:, list(GRIP_IDX)])
        if masks is not None and self.mask_root is not None:
            d = (self.mask_root / "videos" / "chunk-000" / "cam_high" / f"episode_{self.episode:06d}" / "all_frames")
            for si, name in enumerate(MASK_PROMPTS):
                ref = cv2.imread(str(d / name / f"frame_{t:06d}.png"), cv2.IMREAD_GRAYSCALE)
                if ref is None:
                    continue
                ref, got = ref > 0, masks[si] > 0
                self.det_agree[name] += int(ref.any() == got.any())
                if ref.any() or got.any():
                    self.iou[name].append(float((ref & got).sum() / max(1, (ref | got).sum())))
            self.n_mask += 1

    def report(self):
        e, h, g = (np.degrees(np.concatenate(self.err)), np.degrees(np.concatenate(self.hold_err)),
                   np.concatenate(self.grip_err))
        names = [JOINT_NAMES[i] for i in ARM_IDX]
        print("\n>>> 离线自检: 预测 chunk 与示教动作的平均绝对误差(度), 对照 = 原地不动", flush=True)
        print("    " + " ".join(f"{n:>7s}" for n in names), flush=True)
        print("模型 " + " ".join(f"{v:7.2f}" for v in e.mean(0)), flush=True)
        print("不动 " + " ".join(f"{v:7.2f}" for v in h.mean(0)), flush=True)
        r = slice(6, 12)   # 右臂 6 个关节(左臂在这个任务里不动)
        out = {"right_arm_mae_deg": float(e[:, r].mean()), "right_arm_hold_mae_deg": float(h[:, r].mean()),
               "gripper_mae": g.mean(0).round(4).tolist(), "n_infer": len(self.err)}
        print(f">>> 右臂: 模型 {out['right_arm_mae_deg']:.2f}° vs 不动 {out['right_arm_hold_mae_deg']:.2f}°; "
              f"夹爪(L, R) 误差 {out['gripper_mae']}", flush=True)
        if self.n_mask:
            for name in MASK_PROMPTS:
                v = np.asarray(self.iou[name])
                out[f"mask_{name}"] = {"detect_agree": f"{self.det_agree[name]}/{self.n_mask}",
                                       "iou_median": float(np.median(v)) if len(v) else None,
                                       "iou_min": float(v.min()) if len(v) else None}
                print(f">>> mask {name}: 与训练 mask 检出一致 {self.det_agree[name]}/{self.n_mask}, "
                      f"IoU 中位 {out[f'mask_{name}']['iou_median']}, 最小 {out[f'mask_{name}']['iou_min']}", flush=True)
        return out


class Console:
    """stdin 唯一读者: 推理中输入 q 回车 = 结束本局; 局间 ask() 只接受提示语出现之后敲的行。"""

    def __init__(self):
        self._q = queue.Queue()
        self.eof = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                self.eof = True
                break
            self._q.put((time.monotonic(), line.strip()))

    def stop_requested(self):
        hit = False
        while True:
            try:
                _, line = self._q.get_nowait()
            except queue.Empty:
                return hit
            hit |= line.lower() in ("q", "stop")

    def ask(self, prompt, default=""):
        t0 = time.monotonic()
        print(prompt, end="", flush=True)
        while True:
            try:
                ts, line = self._q.get(timeout=0.5)
            except queue.Empty:
                if self.eof:
                    print(f"[stdin 已关闭, 取 '{default}']", flush=True)
                    return default
                continue
            if ts >= t0:
                return line


def run_episode(ep, robot, policy, masker, a, console, ep_dir, evaluator):
    chunks, log = [], []
    macro, first, stop_reason = 0, True, None
    macro_dt = (1.0 / FPS) / a.speed_scale
    max_step = np.radians(a.max_step_deg)
    if a.start_pose == "dataset" and robot.realtime and not a.dry_run:
        print(f">>> 移到示教开局位姿({a.start_time:.0f}s)...", flush=True)
        move_to(robot, START_POSE, a.start_time, a.servo_hz if a.servo_hz > 0 else 50)
    last = np.asarray(robot.read_joints(), np.float64)
    if ep_dir is not None:
        (ep_dir / "infer").mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    t_motion = None
    print(f"\n>>> ===== 第 {ep} 局 ===== (每 {a.chunk} 步重推一次, {macro_dt * 1000:.0f} ms/步, "
          f"dry_run={a.dry_run}); 输入 q 回车 = 结束本局", flush=True)
    try:
        while True:
            if console.stop_requested():
                stop_reason = "q"; break
            if a.steps and macro >= a.steps:
                stop_reason = "steps"; break
            if a.timeout and time.monotonic() - t_start > a.timeout:
                stop_reason = "timeout"; break
            if robot.done():
                stop_reason = "replay_end"; break

            if macro % a.chunk == 0:
                to = time.monotonic()
                obs, masks = get_observation(robot, masker, a.prompt)
                dt_obs = time.monotonic() - to
                if not _jit_done[0]:
                    print(">>> 第一次推理要 JIT 编译, 可能 1-2 分钟(之后每次几百 ms)...", flush=True)
                ti = time.monotonic()
                cw = np.asarray(policy.infer(obs)["actions"], np.float32)   # (10, 14) 绝对关节角 + 夹爪
                dt_inf = time.monotonic() - ti
                _jit_done[0] = True
                if first:
                    first = False
                    if cw.shape[1] != 14:
                        raise RuntimeError(f"模型输出 {cw.shape}, 应为 (H, 14)")
                chunks.append((macro, cw))
                chunks = chunks[-max(2, -(-cw.shape[0] // a.chunk)):]
                st = obs["observation/state"]
                cover = ((masks.reshape(len(masks), -1).mean(1) * 100).astype(float).round(1).tolist()
                         if masks is not None else None)
                log.append({"macro": macro, "t": time.monotonic() - t_start, "state": st, "chunk": cw,
                            "mask_cover_pct": cover, "infer_ms": dt_inf * 1000, "obs_ms": dt_obs * 1000})
                if evaluator is not None:
                    evaluator.add(robot.t, cw, masks)
                if ep_dir is not None:
                    vis = obs["observation/cam_high"][:, :, ::-1].copy()
                    if masks is not None:
                        for m, c in zip(masks, ((0, 255, 0), (255, 0, 255))):   # pitcher 绿, cup 品红
                            vis[m > 0] = (0.45 * vis[m > 0] + 0.55 * np.asarray(c)).astype(np.uint8)
                    cv2.imwrite(str(ep_dir / "infer" / f"{macro:05d}_cam_high.jpg"), vis)
                deg = lambda x: np.degrees(np.asarray(x, np.float64)).round(1).tolist()  # noqa: E731
                print(f">>> [步 {macro}] 取观测{'+mask' if masks is not None else ''} {dt_obs * 1000:.0f} ms, "
                      f"推理 {dt_inf * 1000:.0f} ms"
                      + (f"  mask(pitcher, cup) {cover}%" if cover is not None else "")
                      + f"  右臂 j1-6(度) {deg(st[7:13])} -> 块末 {deg(cw[-1, 7:13])}"
                        f"  右夹爪 {st[13]:.2f} -> {cw[-1, 13]:.2f}", flush=True)

            target = ensemble(chunks, macro, a.ensemble_m)
            if target is None:
                stop_reason = "no_action"; break
            target, clipped = safety(target, last, max_step)
            if clipped:
                print(f"    ⚡ [步 {macro}] 关节单步超过 {a.max_step_deg}°, 已限幅", flush=True)
            if t_motion is None:
                t_motion = time.monotonic()
            if a.dry_run:
                if robot.realtime:
                    time.sleep(macro_dt)
            elif servo(robot, last, target, macro_dt, a.servo_hz, console):
                stop_reason = "q"; break
            last = target
            robot.advance()
            macro += 1
    except KeyboardInterrupt:
        stop_reason = "ctrl-c"
    finally:
        if robot.realtime and not a.dry_run:
            robot.hold()
    rec = {"episode": ep, "config": a.config_name, "checkpoint": str(a.checkpoint), "stop_reason": stop_reason,
           "macro_steps": macro, "n_infer": len(log),
           "motion_time_s": round(time.monotonic() - t_motion, 2) if t_motion else None}
    if ep_dir is not None and log:
        np.savez(ep_dir / "log.npz", macro=np.array([r["macro"] for r in log]), t=np.array([r["t"] for r in log]),
                 state=np.stack([r["state"] for r in log]), actions=np.stack([r["chunk"] for r in log]),
                 infer_ms=np.array([r["infer_ms"] for r in log]), obs_ms=np.array([r["obs_ms"] for r in log]),
                 mask_cover_pct=np.array([r["mask_cover_pct"] or [] for r in log], dtype=object))
    print(f">>> 本局结束({stop_reason}): {macro} 步, 推理 {len(log)} 次, "
          f"平均推理 {np.mean([r['infer_ms'] for r in log[1:]]) if len(log) > 1 else float('nan'):.0f} ms", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="hqi 倒水模型关节空间 rollout")
    ap.add_argument("--config-name", required=True, choices=CONFIGS)
    ap.add_argument("--checkpoint", required=True, help=".../<config>/<exp>/19999(含 params/ 和 assets/)")
    ap.add_argument("--prompt", default=PROMPT, help="必须与训练 tasks.jsonl 逐字一致, 一般不用改")
    ap.add_argument("--robot", choices=["replay", "real"], required=True)
    ap.add_argument("--replay-root", default="~/pi_data/hqi_pour50", help="[replay] LeRobot 数据集目录")
    ap.add_argument("--replay-episode", type=int, default=0)
    ap.add_argument("--replay-mask-root", default=None,
                    help="[replay, mask/ControlVLA] 训练 mask 目录(…/hqi_pour50_masks/hqi_pour50), 给了就顺便比对实时 mask")
    ap.add_argument("--chunk", type=int, default=10, help="每隔几步重新推理; 10 = 整块开环执行完再推(我们真机评测都用 10)")
    ap.add_argument("--ensemble-m", type=float, default=0.3, help="时序集成衰减, 只在 --chunk < 10 有重叠时起作用")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="执行速度倍率, 1.0 = 示教原速(别超过 1)")
    ap.add_argument("--servo-hz", type=float, default=100.0, help="每步内插值下发频率; 0 = 每步只发一次目标")
    ap.add_argument("--max-step-deg", type=float, default=20.0, help="单步(0.1 s)关节位移上限, 示教最大 17°")
    ap.add_argument("--start-pose", choices=["none", "dataset"], default="none",
                    help="dataset = 每局开始前慢慢移到示教第 0 帧的中位位姿(START_POSE)")
    ap.add_argument("--start-time", type=float, default=4.0, help="移到开局位姿用的时间(s)")
    ap.add_argument("--steps", type=int, default=None, help="每局最多多少步(10 步 = 1 s)")
    ap.add_argument("--timeout", type=float, default=None, help="每局最长秒数")
    ap.add_argument("--dry-run", action="store_true", help="[real] 只读传感器 + 推理 + 打印, 不发指令")
    ap.add_argument("--record", default=None, help="会话目录, 缺省 data/runs/<时间>_<config>/")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    if a.speed_scale > 1.0:
        sys.exit("--speed-scale 不要超过 1.0")

    session_dir = None
    if not a.no_record:
        session_dir = (pathlib.Path(a.record).expanduser() if a.record
                       else _ROOT / "data" / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{a.config_name}")
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"))
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f">>> 会话记录: {session_dir}", flush=True)

    print(f">>> 加载模型 {a.config_name} ...", flush=True)
    cfg, policy = load_policy(a.config_name, a.checkpoint, a.prompt)
    masker = make_masker(cfg)
    if a.robot == "replay":
        robot = ReplayRobot(a.replay_root, a.replay_episode)
    else:
        robot = RealRobot(a)
    console = Console()
    results = []
    try:
        if a.robot == "replay":
            ev = ReplayEval(robot, a.replay_mask_root, a.replay_episode)
            rec = run_episode(0, robot, policy, masker, a, console, session_dir / "ep000" if session_dir else None, ev)
            rec["replay_eval"] = ev.report()
            results.append(rec)
            if session_dir is not None:
                (session_dir / "replay_eval.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            return
        ep = 0
        while True:
            rec = run_episode(ep, robot, policy, masker, a, console,
                              session_dir / f"ep{ep:03d}" if session_dir else None, None)
            # 人工判定: 成功 y / 失败 n / 作废 x(操作失误、场景摆错等, 不计入成功率)
            while (v := console.ask(">>> 本局: 成功 y / 失败 n / 作废 x: ", default="x").lower()) not in ("y", "n", "x"):
                pass
            rec["human_success"] = {"y": True, "n": False}.get(v)
            rec["counted"] = v != "x"
            note = console.ask(">>> 备注(回车 = 无): ")
            if note:
                rec["note"] = note
            results.append(rec)
            if session_dir is not None:
                with open(session_dir / "episodes.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            while (ans := console.ask(f">>> 复位场景后输入 y 开始第 {ep + 1} 局, n = 结束: ", default="n").lower()) \
                    not in ("y", "n"):
                pass
            if ans == "n":
                break
            ep += 1
    finally:
        counted = [r for r in results if r.get("counted")]
        if counted:
            ok = sum(bool(r["human_success"]) for r in counted)
            print(f"\n>>> 会话小结: {len(results)} 局, 计入 {len(counted)} 局, 成功 {ok}/{len(counted)}", flush=True)
        robot.close()


if __name__ == "__main__":
    main()
