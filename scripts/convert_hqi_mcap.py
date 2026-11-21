"""hqi 的双臂原始 MCAP 数据 -> LeRobot v2.1 数据集(与我们 AIRBOT 打包数据同一格式: h264, 10 fps, 每 2 帧一个关键帧)。

数据(hqi 2026-11-18 发来的 data1.zip / data2.zip, 共 50 条, 同一个倒水任务):
  /robot/arm_{left,right}_leader/joint_states   主臂 position(6 关节 + 夹爪 0..1), ~180 Hz  -> actions (14)
  /robot/arm_{left,right}_follow/joint_states   从臂 position/velocity/effort, ~180 Hz      -> state (14), velocity, effort
  /diagnostics/frame_timing                      三个相机每帧的采集时间戳(与视频帧一一对应, 与关节 header.stamp 同一时钟)
  videos/cam_high.mp4(俯视 960x540) cam_left.mp4 / cam_right.mp4(左/右手腕 848x480), ~60 fps, 偶有丢帧
14 维顺序 = [左 j1..j6, 左夹爪, 右 j1..j6, 右夹爪](与 hqi 说明一致: 左右各 6 关节 + 1 夹爪, 绝对关节角)。

对齐: 取三路相机和四路关节都覆盖的时间段, 按 10 Hz 等间隔取采样时刻 t_k; 每路相机取采集时间最近的一帧
(按时间戳, 不按帧号, 因为有丢帧); 关节在 t_k 处线性插值。动作 = 主臂在 t_k 的位置(绝对关节角),
训练时由 data config 转成相对 state 的增量(夹爪保持绝对), 模型输出再还原成绝对关节角。

/task_info 里的任务描述(取离心管/拧盖)与视频内容(倒水)不符, 按用户决定统一用 --task 的改正文本。

用法(需要 mcap + mcap-ros2-support + opencv + pyarrow, 例如 scratchpad 的 mcapenv):
  python -I scripts/convert_hqi_mcap.py --src ~/pi_data/hqi_mcap_1118 --out ~/pi_data/hqi_pour50
"""
import argparse
import glob
import json
import multiprocessing as mp
import pathlib
import subprocess

import cv2
import numpy as np

FPS = 10
CAMS = {"cam_high": "cam_high", "cam_left_wrist": "cam_left", "cam_right_wrist": "cam_right"}   # LeRobot key -> 源相机名
ARMS = ("arm_left", "arm_right")
N_IMG_STAT_FRAMES = 10


def read_mcap(path):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    joints = {f"{a}_{r}": ([], [], [], []) for a in ARMS for r in ("leader", "follow")}   # t, pos, vel, eff
    timing, task = {}, None
    with open(path, "rb") as f:
        for _, ch, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            t = ch.topic
            if t == "/task_info":
                task = json.loads(m.data)
            elif t == "/diagnostics/frame_timing":
                j = json.loads(m.data)
                timing[j["role"]] = np.asarray(j["capture_timestamp_ns"], np.int64)
            elif t.endswith("/joint_states"):
                k = t.split("/")[2]
                if list(m.name) != ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]:
                    raise ValueError(f"{path} {t}: 关节名 {list(m.name)}")
                ts, pos, vel, eff = joints[k]
                ts.append(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)
                pos.append(m.position)
                vel.append(m.velocity if len(m.velocity) else [np.nan] * 7)
                eff.append(m.effort if len(m.effort) else [np.nan] * 7)
    out, dup = {}, {}
    for k, (ts, pos, vel, eff) in joints.items():
        ts, pos = np.asarray(ts, np.int64), np.asarray(pos, np.float64)
        vel, eff = np.asarray(vel, np.float64), np.asarray(eff, np.float64)
        if len(ts) < 2 or np.any(np.diff(ts) < 0):
            raise ValueError(f"{path} {k}: 时间戳倒退或为空")
        # 每路 ~6000 条里有 0~29 条与上一条 header.stamp 完全相同(从不倒退): 同一时间戳只留最后一条
        keep = np.append(np.diff(ts) > 0, True)
        d = ~keep
        dup[k] = (int(d.sum()), float(np.abs(pos[np.flatnonzero(d)] - pos[np.flatnonzero(d) + 1]).max()) if d.any() else 0.0)
        out[k] = (ts[keep], pos[keep], vel[keep], eff[keep])
    return out, timing, task, dup


def interp(ts, x, tq):
    return np.stack([np.interp(tq, ts, x[:, i]) for i in range(x.shape[1])], axis=1)


def encode_selected(src_mp4, idxs, dst_mp4):
    """顺序解码 src, 只保留 idxs(可重复)对应的帧, 用 libx264(-g 2, 与我们打包数据一致)写成 10 fps。"""
    cap = cv2.VideoCapture(str(src_mp4))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    need = {}
    for j, i in enumerate(idxs):
        need.setdefault(int(i), []).append(j)
    frames = [None] * len(idxs)
    i, last = 0, max(need)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            raise ValueError(f"{src_mp4}: 读到第 {i} 帧失败(需要到 {last})")
        for j in need.get(i, ()):
            frames[j] = f
        i += 1
    cap.release()
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "23",
                           str(dst_mp4)], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg 失败: {dst_mp4}")
    sel = [frames[k] for k in np.linspace(0, len(frames) - 1, min(N_IMG_STAT_FRAMES, len(frames))).astype(int)]
    a = np.stack([s[:, :, ::-1].astype(np.float32) / 255.0 for s in sel])
    f3 = lambda v: [[[float(x)]] for x in v]  # noqa: E731
    stats = {"min": f3(a.min((0, 1, 2))), "max": f3(a.max((0, 1, 2))), "mean": f3(a.mean((0, 1, 2))),
             "std": f3(a.std((0, 1, 2))), "count": [len(sel)]}
    return (h, w), stats


def convert_episode(args):
    src_dir, ep_idx, out = args
    src_dir, out = pathlib.Path(src_dir), pathlib.Path(out)
    mcap = sorted(src_dir.glob("*.mcap"))
    assert len(mcap) == 1, (src_dir, mcap)
    joints, timing, task, dup = read_mcap(mcap[0])
    tl = json.loads((src_dir / "videos" / "source_timeline.json").read_text())
    for cam in CAMS.values():
        if len(timing[cam]) != tl["cameras"][cam]["frame_count"]:
            raise ValueError(f"{src_dir} {cam}: frame_timing {len(timing[cam])} 帧, timeline {tl['cameras'][cam]['frame_count']}")
    t0 = max([timing[c][0] for c in CAMS.values()] + [v[0][0] for v in joints.values()])
    t1 = min([timing[c][-1] for c in CAMS.values()] + [v[0][-1] for v in joints.values()])
    n = int((t1 - t0) // (10**9 // FPS)) + 1
    tq = t0 + np.arange(n, dtype=np.int64) * (10**9 // FPS)

    def arm14(role, which):
        return np.concatenate([interp(joints[f"{a}_{role}"][0], joints[f"{a}_{role}"][which], tq) for a in ARMS], axis=1)

    state, actions = arm14("follow", 1), arm14("leader", 1)
    velocity, effort = arm14("follow", 2), arm14("follow", 3)
    vids, img_stats, sizes, offs = {}, {}, {}, {}
    for key, cam in CAMS.items():
        ct = timing[cam]
        j = np.clip(np.searchsorted(ct, tq), 1, len(ct) - 1)
        idx = np.where(np.abs(ct[j - 1] - tq) <= np.abs(ct[j] - tq), j - 1, j)
        offs[key] = float(np.abs(ct[idx] - tq).max() / 1e6)
        dst = out / "videos" / "chunk-000" / key / f"episode_{ep_idx:06d}.mp4"
        sizes[key], img_stats[key] = encode_selected(src_dir / "videos" / f"{cam}.mp4", idx, dst)
    cols = {
        "state": state.astype(np.float32), "actions": actions.astype(np.float32),
        "velocity": velocity.astype(np.float32), "effort": effort.astype(np.float32),
        "timestamp": (np.arange(n) / FPS).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep_idx, np.int64),
    }
    src = {"dir": str(src_dir), "mcap": mcap[0].name, "task_info": task, "t0_ns": int(t0), "n": n,
           "max_cam_offset_ms": offs, "image_hw": {k: list(v) for k, v in sizes.items()},
           "dup_stamps": {k: {"n": v[0], "max_pos_diff": v[1]} for k, v in dup.items()},
           "lead_follow_max_abs": float(np.abs(actions - state)[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].max())}
    return ep_idx, cols, img_stats, src


def vec_stats(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [int(len(x))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=pathlib.Path, help="解压目录, 其下 */<数字>/ 是一条")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--task", default="pour the orange juice from the pitcher into the glass cup")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    eps = sorted((pathlib.Path(p) for p in glob.glob(str(a.src.expanduser() / "*" / "*" / "*"))
                  if pathlib.Path(p).is_dir() and pathlib.Path(p).name.isdigit()), key=lambda p: int(p.name))
    names = [int(p.name) for p in eps]
    if names != list(range(len(eps))):
        raise ValueError(f"episode 编号不连续: {names}")
    out = a.out.expanduser()
    if out.exists():
        raise SystemExit(f"{out} 已存在, 先确认再删")
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)

    with mp.Pool(a.workers) as pool:
        results = sorted(pool.map(convert_episode, [(str(p), i, str(out)) for i, p in enumerate(eps)]), key=lambda r: r[0])

    hw = results[0][3]["image_hw"]
    seq = lambda d: {"feature": {"dtype": d, "_type": "Value"}, "length": 14, "_type": "Sequence"}  # noqa: E731
    val = lambda d: {"dtype": d, "_type": "Value"}  # noqa: E731
    hf_feats = {"state": seq("float32"), "actions": seq("float32"), "velocity": seq("float32"), "effort": seq("float32"),
                "timestamp": val("float32"), "frame_index": val("int64"), "episode_index": val("int64"),
                "index": val("int64"), "task_index": val("int64")}
    vec14 = pa.list_(pa.float32(), 14)
    schema = pa.schema([("state", vec14), ("actions", vec14), ("velocity", vec14), ("effort", vec14),
                        ("timestamp", pa.float32()), ("frame_index", pa.int64()), ("episode_index", pa.int64()),
                        ("index", pa.int64()), ("task_index", pa.int64())],
                       metadata={"huggingface": json.dumps({"info": {"features": hf_feats}})})
    start, ep_lines, stat_lines, manifest = 0, [], [], {}
    for ep_idx, cols, img_stats, src in results:
        n = len(cols["state"])
        if src["image_hw"] != hw:
            raise ValueError(f"episode {ep_idx} 图像尺寸 {src['image_hw']} 与第 0 条 {hw} 不同")
        cols["index"] = np.arange(start, start + n, dtype=np.int64)
        cols["task_index"] = np.zeros(n, np.int64)
        arrays = []
        for f in schema:
            v = cols[f.name]
            if pa.types.is_fixed_size_list(f.type):
                arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(v, np.float32).reshape(-1)), 14))
            else:
                arrays.append(pa.array(v))
        pq.write_table(pa.Table.from_arrays(arrays, schema=schema), out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        st = dict(img_stats)
        for k in ("state", "actions", "velocity", "effort", "timestamp", "frame_index", "episode_index", "index", "task_index"):
            st[k] = vec_stats(cols[k])
        ep_lines.append({"episode_index": ep_idx, "tasks": [a.task], "length": n})
        stat_lines.append({"episode_index": ep_idx, "stats": st})
        manifest[ep_idx] = src
        start += n

    vid = lambda h, w: {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],  # noqa: E731
                        "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                 "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    v14 = lambda name: {"dtype": "float32", "shape": [14], "names": [name]}  # noqa: E731
    scal = lambda d: {"dtype": d, "shape": [1], "names": None}  # noqa: E731
    info = {"codebase_version": "v2.1", "robot_type": "hqi_bimanual", "total_episodes": len(results), "total_frames": start,
            "total_tasks": 1, "total_videos": len(results) * len(CAMS), "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
            "splits": {"train": f"0:{len(results)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {**{k: vid(*hw[k]) for k in CAMS}, "state": v14("state"), "actions": v14("actions"),
                         "velocity": v14("velocity"), "effort": v14("effort"), "timestamp": scal("float32"),
                         "frame_index": scal("int64"), "episode_index": scal("int64"), "index": scal("int64"),
                         "task_index": scal("int64")}}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (out / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": a.task}) + "\n")
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ep_lines))
    (out / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(x) + "\n" for x in stat_lines))
    (out / "convert_manifest.json").write_text(json.dumps({"task": a.task, "fps": FPS, "episodes": manifest},
                                                          indent=1, ensure_ascii=False))
    offs = np.array([[m["max_cam_offset_ms"][k] for k in CAMS] for m in manifest.values()])
    print(f"{len(results)} episodes, {start} frames -> {out}")
    print(f"相机帧与采样时刻的最大偏差 ms(各路 max): {dict(zip(CAMS, offs.max(0).round(1).tolist()))}")
    print(f"每条帧数: min {min(x['length'] for x in ep_lines)} max {max(x['length'] for x in ep_lines)}")
    dn = sum(v["n"] for m in manifest.values() for v in m["dup_stamps"].values())
    dd = max(v["max_pos_diff"] for m in manifest.values() for v in m["dup_stamps"].values())
    print(f"重复时间戳: 共 {dn} 条(已去重), 与相邻同时间戳消息的最大位置差 {dd:.4f} rad")
    print(f"主臂-从臂最大关节差(各条 max 的 max): {max(m['lead_follow_max_abs'] for m in manifest.values()):.3f} rad")
    print(f"图像尺寸: {hw}")


if __name__ == "__main__":
    main()
