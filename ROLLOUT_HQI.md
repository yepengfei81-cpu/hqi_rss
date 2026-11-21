# Rollout guide for the pouring models

This guide explains how to run the three pouring models on your robot, and how we do rollouts on AIRBOT. All three models were trained on the 50 pouring demonstrations you sent; the configs are baseline, mask and ControlVLA. The code is `rollout_hqi_pour.py` in this repo. It is robot-agnostic, so you need to connect your own hardware.

Suggested order:

1. Copy the weights, data and code to the robot PC and set up the environment (Section 3).
2. Without the robot, run the offline self-check (Section 4). It feeds one demonstration to the model as if it were the robot, to confirm the outputs are sensible and the live masks match the training masks.
3. Implement the 5 methods of `RealRobot` in the script (Section 5). Run with `--dry-run` first (inference only, no motion), then for real.

---

## 1. Where the weights and code are

The weights and data are on the cluster: `P=/n/netscratch/hankyang_lab/Lab/pengfei`, with the weights under `$P/openpi_hqi_pour/checkpoints/`. The code in this repo is the same as `$P/openpi_hqi_pour` on the cluster, which is what trained the three models.

| Method | `--config-name` | Weights (final, 20k steps) |
|---|---|---|
| baseline | `pi05_hqi_pour50` | `checkpoints/pi05_hqi_pour50/pour50_base_v1/19999` |
| mask (your mask prompt + MCR) | `pi05_hqi_pour50_mask` | `checkpoints/pi05_hqi_pour50_mask/pour50_mask_v1/19999` |
| ControlVLA | `pi05_hqi_pour50_controlvla` | `checkpoints/pi05_hqi_pour50_controlvla/pour50_controlvla_v1/19999` |

- Each `19999/` has three subdirectories: `params/` (6.0 GB) holds the weights, `assets/` the normalization statistics, and `train_state/` (3 GB) the optimizer state. Deployment needs only `params/` and `assets/`; you do not need to copy `train_state/`.
- All three models were trained on the same data with the same hyperparameters: π0.5, 2 GPUs, batch 48, 20k steps. They differ only in the mask method, and their training losses nearly coincide.
- netscratch purges files after 90 days, so please copy the weights off soon.

Related code:

| File | Contents |
|---|---|
| `rollout_hqi_pour.py` | The rollout script described here: offline self-check, real-robot skeleton, session logging |
| `airbot/hf_mask.py` | Mask generator. Training masks and deployment masks come from this same file |
| `scripts/gen_head_masks.py` | The command that generated the training masks, in the file header |
| `src/openpi/policies/hqi_joint_policy.py` | Mapping from observation keys to model inputs |
| `src/openpi/training/config.py` | Search for `pi05_hqi_pour50` to find the three configs and `LeRobotHqiJointDataConfig` |
| `scripts/convert_hqi_mcap.py` | How your MCAP files were converted to training data, including the 10 Hz alignment |
| `train_hqi_pour.sh` | The sbatch script used to train the three models on the cluster |
| `reference/local_inference_eef.py` | Our own AIRBOT rollout script, for reading only; see Section 7 |

## 2. Model input/output contract

All of these must match training. The script already follows them.

| Item | Contract |
|---|---|
| prompt | `pour the orange juice from the pitcher into the glass cup`, verbatim. The `/task_info` in the data ("取离心管/拧盖", pick up a centrifuge tube / unscrew a cap) is wrong; training replaced it with this sentence everywhere |
| images | `observation/cam_high` is the top-down camera, 960×540; `observation/cam_left_wrist` and `observation/cam_right_wrist` are the wrist cameras, 848×480. Format is `uint8` **RGB** (HWC), and they must be the same cameras in the same mounting positions as during collection. The model resizes with padding to 224×224 internally, so do not resize yourself |
| state | `observation/state`, 14-dim float32, the current position of the **follower** arms. Order is `[left j1..j6, left gripper, right j1..j6, right gripper]`, joints in radians, grippers in 0..1, same convention as the recordings |
| output | `policy.infer(obs)["actions"]`, shape (10, 14), same layout as state, **absolute** positions. Each step is 0.1 s (10 Hz), so one chunk is 1 second. In training, the action is the **leader** arm position at the same instant, so think of the output as "the model replacing the leader arm" |
| internal processing | During training the 6 joints of each arm are converted to deltas relative to the current state; grippers stay absolute. Normalization statistics are read from the checkpoint's `assets/`. The policy does all of this automatically, and the output is already converted back to absolute |
| mask | Only the mask and ControlVLA models need it. Key `observation/base_region_mask`, shape (2, H, W), float32 0/1, slot order `pitcher`, `cup`, top-down camera only. No MCR at inference; images stay unchanged |

How the training data was sampled: timestamps evenly spaced at 10 Hz over the raw MCAP. At each timestamp, each camera uses the frame with the nearest capture time (by timestamp, not frame index, because frames are sometimes dropped), and joints are linearly interpolated.

**The masks must use exactly the training settings.** The `MASK_*` constants at the top of `rollout_hqi_pour.py` are these settings:

- Detector: HF `IDEA-Research/grounding-dino-base` + `facebook/sam-vit-base`.
- Detection phrases: `orange juice pitcher` and `small glass cup`.
- Threshold 0.2.
- Slots are assigned by box area, not by phrase: the pitcher covers 2.5%–15% of the image, the cup less than 2.5%. Both are transparent containers and DINO cannot tell which is which, so assigning by phrase swaps them.
- The long side of the image is resized to 512 before detection.

## 3. Environment

```bash
# Code comes from this repo; the three weights (without train_state, about 18 GB) and the self-check data and masks (about 1.1 GB) come from the cluster
git clone git@github.com:yepengfei81-cpu/hqi_rss.git ~/openpi_hqi_pour
P=/n/netscratch/hankyang_lab/Lab/pengfei
rsync -r --exclude train_state <cluster>:$P/openpi_hqi_pour/checkpoints/ ~/openpi_hqi_pour/checkpoints/
rsync -r <cluster>:$P/pi_data/hqi_pour50 <cluster>:$P/pi_data/hqi_pour50_masks ~/pi_data/

cd ~/openpi_hqi_pour
GIT_LFS_SKIP_SMUDGE=1 uv sync
# Required: switch the transformers used by the masks to the version that generated the training masks (reason below)
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@b6fb4ec0f24cfad232de46a02441b9cc6362cd22" \
    huggingface_hub==1.26.0 tokenizers==0.22.2 regex==2025.11.3
```

**Why change transformers:** openpi's lock file pins 4.53.2, but the training masks were generated with 5.8.0.dev0, the commit above. We compared 63 frames using the same code:

| Environment | Agreement with training masks |
|---|---|
| 4.53.2 | Pitcher: 4 frames with IoU < 0.9 (lowest 0.06). Cup: 3 frames disagree, each detected on one side and missed on the other |
| The versions above (uv also pulls in httpx and other dependencies); everything else from openpi's original lock | 63/63 frames pixel-identical, IoU 1.0 on all |

Also, under 4.53.2 `hf_mask.py` originally crashed, because `from_pretrained` does not accept the `dtype` argument. We added a compatibility fallback, but that alone does not make the masks match, so you still need to change the version.

Notes:

- After the change, run with `.venv/bin/python`, **not `uv run`**. By default `uv run` re-syncs to the lock file and downgrades transformers back to 4.53.2.
- The first run needs network access to download:
  - DINO and SAM, about 1.25 GB in total, into `~/.cache/huggingface`;
  - the PaliGemma tokenizer, into `~/.cache/openpi`.
- After downloading, we recommend setting `HF_HUB_OFFLINE=1`. If the network or DNS fails on the robot, HF exits with an error instead of falling back to the local cache.
- GPU memory: π0.5 inference uses about 6.6 GB; with DINO+SAM the whole GPU uses about 9.3 GB. We run both together on a 12 GB GPU. For the mask and ControlVLA configs, the script automatically stops JAX from preallocating memory (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `ALLOCATOR=platform`); otherwise DINO runs out of memory.

## 4. Step 1: offline self-check (no robot)

```bash
cd ~/openpi_hqi_pour
export HF_LEROBOT_HOME=~/pi_data HF_HUB_OFFLINE=1
for c in pi05_hqi_pour50:pour50_base_v1 pi05_hqi_pour50_mask:pour50_mask_v1 pi05_hqi_pour50_controlvla:pour50_controlvla_v1; do
  cfg=${c%%:*}; exp=${c##*:}
  .venv/bin/python rollout_hqi_pour.py --config-name $cfg --checkpoint checkpoints/$cfg/$exp/19999 \
      --robot replay --replay-root ~/pi_data/hqi_pour50 --replay-episode 3 \
      --replay-mask-root ~/pi_data/hqi_pour50_masks/hqi_pour50
done
```

What this does:

- It treats the three videos and the state of demonstration 3 as sensor readings and runs inference every 10 steps (1 second).
- It compares the 10 predicted steps with the next 10 leader-arm actions in the demonstration, and finally prints the mean error per joint (degrees). As a reference it also prints the error of "holding still" (the row labeled 不动).
- For the mask and ControlVLA models, it also compares each live mask with the training mask.

How to read the results (the script prints in Chinese):

- **Mask agreement**: it should print `检出一致 N/N` (detection agreement N/N) and `IoU 中位 1.0` (median IoU 1.0). If it is clearly lower, the environment is wrong; switch transformers as in Section 3 first.
- **Action error**: the right-arm error should be clearly smaller than the hold-still row. This is open-loop prediction on the training set, so the numbers only confirm the pipeline is wired correctly (for example, RGB/BGR swapped, wrong joint order or wrong prompt). They do not predict real-robot performance.

What we have already tested: the real input/output transforms, real mask generation and the model's image preprocessing all run end to end, with only the network replaced by a fake one (we did not run the real weights). For all three configs, the observations reach the model as specified: all three images are 224×224, the mask is (2, 224, 224) and nonzero only on the top-down camera, and the state is padded to 32 dims. Live masks match the training masks: the ControlVLA self-check covered all of demonstration 15, with IoU 1.0 on 33/33 frames. We also ran the real-time path with a fake robot: interpolated commands, emergency stop, human judgment and the log files all work. **The self-check numbers with the real weights have not been run yet**; please send them to us once you have them.

## 5. Step 2: connect your robot

Implement the 5 methods of `RealRobot` in `rollout_hqi_pour.py`. The runner touches the hardware only through these 5 methods:

| Method | Requirement |
|---|---|
| `read_images()` | Return `{cam_high, cam_left_wrist, cam_right_wrist}`, each an (H, W, 3) uint8 RGB image. OpenCV reads BGR, so convert it |
| `read_joints()` | Return the current follower position, 14 dims, i.e. the position in `/robot/arm_{left,right}_follow/joint_states` |
| `command_joints(q)` | Send a 14-dim target position, non-blocking. The simplest approach is to treat q as the leader position and send it through the same "leader → follower" control path as teleoperation |
| `hold()` | Stop at the current position. Called at the end of each episode and on emergency stop |
| `close()` | Release resources |

Then bring it up on the robot in this order:

```bash
# 1) Read sensors, run inference and print, without moving the robot. Check that the printed right-arm targets are sensible and the mask coverage looks normal (top-down view: pitcher about 2%, cup about 0.7%)
.venv/bin/python rollout_hqi_pour.py --config-name pi05_hqi_pour50 --checkpoint checkpoints/pi05_hqi_pour50/pour50_base_v1/19999 \
    --robot real --dry-run --steps 50
# 2) Real motion. You can start slow with --speed-scale 0.5; type q and Enter in the terminal to end the episode (Ctrl+C also works)
.venv/bin/python rollout_hqi_pour.py --config-name pi05_hqi_pour50 --checkpoint checkpoints/pi05_hqi_pour50/pour50_base_v1/19999 \
    --robot real --start-pose dataset
```

Common options:

- `--start-pose dataset`: before each episode, move slowly over 4 seconds to the median frame-0 pose of the 50 demonstrations. This pose is essentially the zero position with both grippers open. Matching the demonstrations' starting pose matters.
- `--servo-hz 100`: within each step (0.1 s), interpolate linearly and send at 100 Hz. If your follower controller interpolates on its own, set this to 0 to send only once per step.
- `--max-step-deg 20`: per-step joint displacement limit, to prevent wild jumps. The largest single right-arm step in the demonstrations is 17°. Grippers are also clamped to 0..1.
- `--chunk 10` (default): execute all 10 steps of a chunk open loop before inferring the next one. We used 10 for all real-robot evaluations, because discrete events such as releasing the gripper are not cut off midway. A smaller value (for example 5) reacts faster, and overlapping chunks are averaged with time-decaying weights.
- `--speed-scale`: execution speed multiplier. 1.0 is the demonstration speed; do not exceed 1.

## 6. What the script does

```
per episode:  (optional) move to the start pose
              loop, 0.1 s per step:
                if step % chunk == 0:
                  read the three images → immediately read the follower state   ← keep these as close in time as possible
                  (mask/ControlVLA) run DINO+SAM on the top-down image → masks for the 2 slots
                  policy.infer → (10, 14) absolute targets
                take this step's target → per-step clamp → interpolate and send within 0.1 s
              end (q / step limit / timeout) → hold() → human judgment y/n/x → append to episodes.jsonl
```

- Inference is synchronous: the robot pauses during inference. On our 12 GB GPU, π0.5 takes about 230 ms and a mask frame about 350 ms, so a mask model pauses about 0.6 s per second of motion. All our three-way comparisons on AIRBOT ran this way.
- Read the images first, then the state immediately after, and only then run the masks. If the masks ran in between, the state would be about 0.3 s newer than the images, unlike training, where image and state come from the same instant.
- The first inference triggers JIT compilation, about 1–2 minutes; after that each inference takes a few hundred milliseconds. This compilation happens at the start of the first episode, before the robot moves.

## 7. How we do rollouts on AIRBOT, for reference

Our own script is `reference/local_inference_eef.py`. It is for reading only and does not run in this repo, because the AIRBOT drivers, camera and gripper-model code it depends on are not included. The main loop is in `EEFRunner._run_episode`, observations in `get_observation`, and interpolated commands in `_servo_macro`.

The main difference from your setup is that it works in end-effector (EEF) space. The model outputs end-effector poses in each arm's own start-pose frame, and we send them with a 250 Hz Cartesian servo. That coordinate transform does not apply to your joint-space models. Everything else is the same: loop structure, timing, masks, logging and evaluation protocol.

These practices came from problems we ran into:

1. **Use the prompt verbatim.** One different word puts the model in a condition it has never seen.
2. **Generate deployment masks with the same file and parameters as the training masks.** Compare them offline before going on the robot. We once found training masks that had not loaded and were silently filled with zeros, so the training loader now enables `strict_masks` and fails on any missing file.
3. **Match the demonstrations' starting pose.** Before each episode we align the follower to the leader's current joint angles (`--start-from-lead`), so the joint configuration matches the demonstrations; otherwise inverse kinematics may pick another joint branch and hit a limit.
4. **When JAX and torch share one GPU**, stop JAX from preallocating memory, or DINO runs out of memory.
5. **Set `HF_HUB_OFFLINE=1` on the robot**, so a network hiccup does not kill the process.
6. **Use per-step clamping and slow trial runs.** For a new model, start with `--dry-run`, then slow speed, then full speed.

**Evaluation protocol** (fair comparison of the three models):

- Each model gets 20 counted episodes. All three run in the same scene with the same range of object placements and identical runtime parameters.
- Right after each episode a person judges it: success y, failure n, void x. Void is only for problems unrelated to the model, such as operator error or a wrongly set-up scene; void episodes are not counted and their records are deleted.
- Each model reports one overall success rate, not split further.
- Timing starts at the first motion command.
- We also save raw top-down camera frames at 2 Hz during each episode, to make videos afterwards.

## 8. Log files

By default logs go to `data/runs/<time>_<config>/`; set a directory with `--record DIR` or turn logging off with `--no-record`.

| File | Contents |
|---|---|
| `session.json` | All parameters of the session |
| `episodes.jsonl` | One line per episode: `human_success`, `counted` (false for void), `stop_reason`, `motion_time_s`, `n_infer`, note |
| `epNNN/infer/NNNNN_cam_high.jpg` | The top-down image at each inference. For mask models the masks are overlaid: pitcher green, cup magenta |
| `epNNN/log.npz` | Per inference: `state`, `actions` (10×14), `infer_ms`, `obs_ms`, mask coverage |
| `replay_eval.json` | Statistics from the offline self-check |
