# 倒水模型 rollout 说明

本文说明三个倒水模型怎么在你的机器人上跑，以及我们在 AIRBOT 上做 rollout 的方式。三个模型都在你发来的 50 条倒水数据上训练，配置分别是 baseline、mask、ControlVLA。代码就是本目录下的 `rollout_hqi_pour.py`，它和机器人无关，硬件部分要你自己接。

建议按这个顺序来：

1. 把权重、数据和代码拷到机器人电脑上，配好环境（第 3 节）。
2. 不连机器人，先跑离线自检（第 4 节）。用一条示教数据当"机器人"喂给模型，确认模型输出合理、实时 mask 和训练 mask 一致。
3. 在脚本里实现 `RealRobot` 的 5 个方法（第 5 节），先用 `--dry-run` 只推理不动，再真动。

---

## 1. 权重和代码在哪

权重和数据在集群上：`P=/n/netscratch/hankyang_lab/Lab/pengfei`，权重在 `$P/openpi_hqi_pour/checkpoints/` 下。这个仓库里的代码和集群上 `$P/openpi_hqi_pour` 的是同一份，三个模型就是用它训练的。

| 方法 | `--config-name` | 权重（20k 步的最终版） |
|---|---|---|
| baseline | `pi05_hqi_pour50` | `checkpoints/pi05_hqi_pour50/pour50_base_v1/19999` |
| mask（你的 mask prompt + MCR） | `pi05_hqi_pour50_mask` | `checkpoints/pi05_hqi_pour50_mask/pour50_mask_v1/19999` |
| ControlVLA | `pi05_hqi_pour50_controlvla` | `checkpoints/pi05_hqi_pour50_controlvla/pour50_controlvla_v1/19999` |

- 每个 `19999/` 下有三个子目录：`params/` 6.0 GB 是权重，`assets/` 里是归一化统计量，`train_state/` 3 GB 是优化器状态。部署只需要 `params/` 和 `assets/`，`train_state/` 不用拷。
- 三个模型用同一份数据、同一组超参训练：π0.5，2 卡，batch 48，20k 步。它们只在 mask 方法上不同，训练 loss 基本重合。
- netscratch 有 90 天清理，权重请尽早拷走。

相关代码：

| 文件 | 内容 |
|---|---|
| `rollout_hqi_pour.py` | 本文说的 rollout 脚本：离线自检、真机骨架、会话记录 |
| `airbot/hf_mask.py` | mask 生成器。训练 mask 和部署 mask 是同一个文件 |
| `scripts/gen_head_masks.py` | 训练 mask 的生成命令，见文件头 |
| `src/openpi/policies/hqi_joint_policy.py` | 观测键到模型输入的映射 |
| `src/openpi/training/config.py` | 搜 `pi05_hqi_pour50` 可以看到三个配置和 `LeRobotHqiJointDataConfig` |
| `scripts/convert_hqi_mcap.py` | 你的 MCAP 怎么转成训练数据，包括 10 Hz 对齐方式 |
| `train_hqi_pour.sh` | 集群上训练这三个模型用的 sbatch 脚本 |
| `reference/local_inference_eef.py` | 我们 AIRBOT 自己的 rollout 脚本，只供阅读，见第 7 节 |

## 2. 模型的输入输出约定

这些都必须和训练一致，脚本里已经按这个写好。

| 项 | 约定 |
|---|---|
| prompt | `pour the orange juice from the pitcher into the glass cup`，要逐字一致。数据里 `/task_info` 写的"取离心管/拧盖"不对，训练时已经统一改成这句 |
| 图像 | `observation/cam_high` 是俯视相机，960×540；`observation/cam_left_wrist` 和 `observation/cam_right_wrist` 是手腕相机，848×480。格式是 `uint8` **RGB**（HWC），必须是采集时的同一台相机、同一机位。模型内部会按比例缩放并补边到 224×224，你不用自己缩放 |
| state | `observation/state`，14 维 float32，取**从臂（follow）**的当前位置。顺序是 `[左 j1..j6, 左夹爪, 右 j1..j6, 右夹爪]`，关节用弧度，夹爪是 0..1，和录制时同一约定 |
| 输出 | `policy.infer(obs)["actions"]`，形状 (10, 14)，布局同 state，是**绝对**位置。每步 0.1 s（10 Hz），一块就是 1 秒。训练时 action 取**主臂（leader）**在同一时刻的位置，所以输出可以理解为"模型代替主臂" |
| 内部处理 | 6 个关节训练时转成相对当前 state 的增量，夹爪保持绝对值。归一化统计量从 checkpoint 的 `assets/` 读。这些 policy 都会自动做，输出已经还原成绝对值 |
| mask | 只有 mask 和 ControlVLA 模型需要。键是 `observation/base_region_mask`，形状 (2, H, W)，float32 0/1，槽位顺序是 `pitcher`、`cup`，只给俯视相机。推理时不做 MCR，图像保持原样 |

训练数据的采样方式：在原始 MCAP 上按 10 Hz 等间隔取时刻。每个时刻每路相机取采集时间最近的一帧（按时间戳，不按帧号，因为有丢帧），关节做线性插值。

**mask 必须和训练用同一套设置**，`rollout_hqi_pour.py` 顶部的 `MASK_*` 常量就是这套设置：

- 检测器：HF `IDEA-Research/grounding-dino-base` + `facebook/sam-vit-base`。
- 检测词：`orange juice pitcher` 和 `small glass cup`。
- 阈值 0.2。
- 槽位按框的面积分配，不按检测词：水壶占整图 2.5%–15%，杯子小于 2.5%。原因是水壶和杯子都是透明容器，DINO 分不清哪个是哪个，按检测词分会把两者对调。
- 推理前先把图像长边缩到 512。

## 3. 环境

```bash
# 代码从这个仓库拿；三份权重（不拷 train_state，约 18 GB）和自检用的数据、mask（约 1.1 GB）从集群拷
git clone git@github.com:yepengfei81-cpu/hqi_rss.git ~/openpi_hqi_pour
P=/n/netscratch/hankyang_lab/Lab/pengfei
rsync -r --exclude train_state <集群>:$P/openpi_hqi_pour/checkpoints/ ~/openpi_hqi_pour/checkpoints/
rsync -r <集群>:$P/pi_data/hqi_pour50 <集群>:$P/pi_data/hqi_pour50_masks ~/pi_data/

cd ~/openpi_hqi_pour
GIT_LFS_SKIP_SMUDGE=1 uv sync
# 必须做：把 mask 用到的 transformers 换成生成训练 mask 时的版本（原因见下面）
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@b6fb4ec0f24cfad232de46a02441b9cc6362cd22" \
    huggingface_hub==1.26.0 tokenizers==0.22.2 regex==2025.11.3
```

**为什么要换 transformers：** openpi 的 lock 文件锁的是 4.53.2，训练 mask 是用 5.8.0.dev0 生成的，也就是上面那个 commit。我用同一段代码抽了 63 帧比较：

| 环境 | 和训练 mask 的一致性 |
|---|---|
| 4.53.2 | 水壶 4 帧 IoU<0.9（最低 0.06）；杯子 3 帧对不上，都是一边检到、一边没检到 |
| 换成上面这组版本（uv 会顺带装上 httpx 等依赖），其余是 openpi 原 lock 的包 | 63/63 帧逐像素一致，IoU 全部是 1.0 |

另外，4.53.2 下 `hf_mask.py` 原来会直接报错，因为 `from_pretrained` 不认 `dtype` 参数。我已经加了兼容写法，但只靠这个改动 mask 仍然对不上，所以还是要换版本。

注意事项：

- 换完以后请用 `.venv/bin/python` 运行，**不要用 `uv run`**。`uv run` 默认会按 lock 重新同步，把 transformers 降回 4.53.2。
- 第一次运行要联网，会下载这几样东西：
  - DINO 和 SAM，共约 1.25 GB，放在 `~/.cache/huggingface`；
  - PaliGemma tokenizer，放在 `~/.cache/openpi`。
- 下载完以后建议设 `HF_HUB_OFFLINE=1`。真机上网络或 DNS 出问题时，HF 会直接报错退出，不会退回去用本地缓存。
- 显存：π0.5 推理实际占约 6.6 GB，加上 DINO+SAM 整卡约 9.3 GB。我们在 12 GB 的卡上同时跑两者。对 mask 和 ControlVLA 配置，脚本会自动设 JAX 不预占显存（`XLA_PYTHON_CLIENT_PREALLOCATE=false`，`ALLOCATOR=platform`），否则 DINO 会 OOM。

## 4. 第一步：离线自检（不连机器人）

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

这一步做的事：

- 把第 3 条示教的三路视频和 state 当作传感器读数，每 10 步（1 秒）推理一次。
- 把预测的 10 步和示教里接下来 10 步的主臂动作比较，最后打印每个关节的平均误差（度），同时给出"原地不动"的误差作为对照。
- 对 mask 和 ControlVLA 模型，再把实时生成的 mask 和训练 mask 逐帧比较。

怎么看结果：

- **mask 一致性**：应该是"检出一致 N/N、IoU 中位 1.0"。如果明显低于这个，说明环境不对，先按第 3 节换 transformers。
- **动作误差**：右臂误差应该明显小于"不动"那一行。这是训练集上的开环预测，数值只用来确认整条链路没接错，比如 RGB/BGR 反了、关节顺序错了或者 prompt 不对，不代表真机效果。

我这边已经测过的部分：真实的输入输出变换、真实的 mask 生成和模型的图像预处理都跑通了，只把网络换成了假的（这边没有用真权重跑）。三个配置的观测都按约定进了模型：三路图像都是 224×224，mask 是 (2, 224, 224) 且只在俯视相机上有值，state 补齐到 32 维。实时 mask 和训练 mask 一致：ControlVLA 那次自检覆盖第 15 条整条示教，33/33 帧 IoU=1.0。实时路径我也用假机器人跑过：插值下发、急停、人工判定和记录文件都正常。**用真权重的自检数字还没跑**，你跑完把结果发我一下。

## 5. 第二步：接你的机器人

在 `rollout_hqi_pour.py` 里实现 `RealRobot` 的 5 个方法。runner 只通过这 5 个方法碰硬件：

| 方法 | 要求 |
|---|---|
| `read_images()` | 返回 `{cam_high, cam_left_wrist, cam_right_wrist}`，每路是 (H, W, 3) uint8 RGB。OpenCV 读出来是 BGR，要转一下 |
| `read_joints()` | 返回从臂当前位置，14 维，即 `/robot/arm_{left,right}_follow/joint_states` 里的 position |
| `command_joints(q)` | 下发 14 维目标位置，不能阻塞。最省事的接法是把 q 当成主臂位置，走遥操作时"主臂 → 从臂"的同一条控制链 |
| `hold()` | 停在当前位置。每局结束或急停时调用 |
| `close()` | 释放资源 |

然后按这个顺序上机：

```bash
# 1) 只读传感器、推理、打印，机器人不动。检查打印出的右臂目标角度合不合理、mask 覆盖率是否正常（俯视下水壶约 2%，杯子约 0.7%）
.venv/bin/python rollout_hqi_pour.py --config-name pi05_hqi_pour50 --checkpoint checkpoints/pi05_hqi_pour50/pour50_base_v1/19999 \
    --robot real --dry-run --steps 50
# 2) 真动。可以先 --speed-scale 0.5 慢速试；终端输入 q 回车 = 结束本局（Ctrl+C 也行）
.venv/bin/python rollout_hqi_pour.py --config-name pi05_hqi_pour50 --checkpoint checkpoints/pi05_hqi_pour50/pour50_base_v1/19999 \
    --robot real --start-pose dataset
```

常用参数：

- `--start-pose dataset`：每局开始前，用 4 秒慢慢移到 50 条示教第 0 帧的中位位姿。这个位姿基本就是零位，两个夹爪张开。开局姿态和示教一致很重要。
- `--servo-hz 100`：每一步（0.1 s）内做线性插值，按 100 Hz 下发。如果你的从臂控制器自己会插值，可以设成 0，表示每步只发一次。
- `--max-step-deg 20`：单步关节位移上限，用来防止野跳。示教里右臂单步最大是 17°。另外夹爪会被限制在 0..1。
- `--chunk 10`（默认）：一块 10 步全部开环执行完再推理下一块。我们真机评测都用 10，因为离散事件（比如松夹爪）不会被中途打断。改小（比如 5）反应更快，重叠部分会按时间加权平均。
- `--speed-scale`：执行速度倍率。1.0 是示教原速，不要超过 1。

## 6. 脚本在做什么

```
每局:  (可选) 移到开局位姿
       循环, 每步 0.1 s:
         若 步数 % chunk == 0:
           取三路图像 → 紧接着读从臂 state      ← 两者要尽量同一时刻
           (mask/ControlVLA) 俯视图跑 DINO+SAM → 2 个槽位的 mask
           policy.infer → (10, 14) 绝对目标
         取当前这一步的目标 → 单步限幅 → 在 0.1 s 内插值下发
       结束(q / 步数 / 超时) → hold() → 人工判定 y/n/x → 写 episodes.jsonl
```

- 推理是同步的：推理期间机器人停住。在我们 12 GB 的卡上，π0.5 一次约 230 ms，mask 一帧约 350 ms，所以 mask 模型大约每走 1 秒停 0.6 秒。我们 AIRBOT 上的三方对比都是这样跑的。
- 先读图、紧接着读 state，再跑 mask。如果 mask 放在两者之间，state 会比图像新约 0.3 秒，和训练里图像与 state 同一时刻的对齐不一致。
- 第一次推理要 JIT 编译，约 1–2 分钟，之后每次几百毫秒。这次编译发生在第一局开始时，机器人还没动。

## 7. 我们在 AIRBOT 上怎么做 rollout，供参考

我们自己的脚本是 `reference/local_inference_eef.py`。它只供阅读，在这个仓库里跑不起来，因为它依赖的 AIRBOT 驱动、相机和夹爪模型代码没有放进来。主循环在 `EEFRunner._run_episode`，观测在 `get_observation`，插值下发在 `_servo_macro`。

它和你的情况有一个主要区别：它是末端空间（EEF）的。模型输出的是每条臂各自开局坐标系里的末端位姿，我们用 250 Hz 的笛卡尔 servo 下发。这部分坐标变换对你的关节空间模型不适用。其余部分是同一套：循环结构、计时、mask、记录和评测方式。

这些做法是踩过坑以后定下来的：

1. **prompt 逐字一致**。差一个词，模型就会进入它没见过的条件。
2. **部署 mask 和训练 mask 用同一个文件、同一组参数**。上线前离线比对一遍：我们之前发现过训练时 mask 没读到、被静默填零的情况，所以训练 loader 现在开了 `strict_masks`，缺文件直接报错。
3. **开局姿态和示教一致**。我们每局开始前把从臂对齐到主臂当前的关节角（`--start-from-lead`），保证关节构型和示教一样，否则逆解可能走另一个关节分支，撞到限位。
4. **JAX 和 torch 共用一张卡**时，要设 JAX 不预占显存，否则 DINO 会 OOM。
5. **真机上设 `HF_HUB_OFFLINE=1`**，避免网络抖一下进程就退出。
6. **单步限幅和慢速试跑**。新模型先 `--dry-run`，再慢速，最后原速。

**评测方式**（三个模型公平对比）：

- 每个模型计入 20 局。三个模型在同一场景、同一套物体摆放范围下跑，运行参数完全相同。
- 每局结束立刻由人判定：成功 y、失败 n、作废 x。作废只用于操作失误或场景摆错这类和模型无关的情况，作废局不计入，记录直接删掉。
- 每个模型最后报一个总成功率，不拆分。
- 计时从第一条运动指令开始。
- 我们还会在局内以 2 Hz 存俯视相机原图，方便事后剪视频。

## 8. 记录文件

缺省写到 `data/runs/<时间>_<config>/`，可以用 `--record DIR` 指定，`--no-record` 关闭。

| 文件 | 内容 |
|---|---|
| `session.json` | 本次所有参数 |
| `episodes.jsonl` | 每局一行：`human_success`、`counted`（作废=false）、`stop_reason`、`motion_time_s`、`n_infer`、备注 |
| `epNNN/infer/NNNNN_cam_high.jpg` | 每次推理时的俯视图。mask 模型会叠加 mask：水壶绿色，杯子品红 |
| `epNNN/log.npz` | 每次推理的 `state`、`actions`（10×14）、`infer_ms`、`obs_ms`、mask 覆盖率 |
| `replay_eval.json` | 离线自检的统计结果 |
