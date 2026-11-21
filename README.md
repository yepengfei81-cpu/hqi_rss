# 双臂倒水：π0.5 三种方法的训练与 rollout 代码

数据是 hqi 采的 50 条双臂倒水示教（MCAP）：右臂把水壶里的橙汁倒进玻璃杯，再把水壶放回。在这份数据上用同一组超参训练了三个模型：

| 方法 | 配置名 |
|---|---|
| baseline | `pi05_hqi_pour50` |
| mask prompt + MCR | `pi05_hqi_pour50_mask` |
| ControlVLA | `pi05_hqi_pour50_controlvla` |

**怎么在机器人上跑，请看 [ROLLOUT_HQI.md](ROLLOUT_HQI.md)**：权重位置、输入输出约定、环境、离线自检、接自己的机器人，以及我们在 AIRBOT 上的 rollout 经验。

## 仓库里有什么

| 路径 | 内容 |
|---|---|
| `rollout_hqi_pour.py` | 关节空间 rollout 脚本，和机器人无关：离线自检、真机骨架（`RealRobot`）、会话记录 |
| `ROLLOUT_HQI.md` | rollout 说明 |
| `airbot/hf_mask.py` | mask 生成器（Grounding DINO + SAM，按框面积分槽位）。训练 mask 和部署 mask 是同一个文件 |
| `scripts/convert_hqi_mcap.py` | 原始 MCAP 转成 LeRobot 数据集（10 Hz） |
| `scripts/gen_head_masks.py` | 生成训练 mask，命令见文件头 |
| `scripts/compute_norm_stats.py`、`scripts/train.py`、`train_hqi_pour.sh` | 归一化统计量、训练入口、集群 sbatch 脚本 |
| `src/openpi/` | 模型和训练代码，三个配置在 `src/openpi/training/config.py` 里搜 `pi05_hqi_pour50` |
| `assets/` | 三个配置的归一化统计量（`norm_stats.json`，三份内容相同；checkpoint 里也各有一份） |
| `reference/local_inference_eef.py` | 我们 AIRBOT 的 rollout 脚本，只供阅读，依赖的硬件代码不在这里 |

## 不在仓库里的东西（在集群上）

集群目录 `P=/n/netscratch/hankyang_lab/Lab/pengfei`：

- 权重：`$P/openpi_hqi_pour/checkpoints/<配置名>/<实验名>/19999`，部署只需要其中的 `params/` 和 `assets/`。
- 训练数据：`$P/pi_data/hqi_pour50`（LeRobot 格式）。
- 训练 mask：`$P/pi_data/hqi_pour50_masks`。

netscratch 有 90 天清理，请尽早拷走。

## 来源

代码基于 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi)，许可证见 `LICENSE`（Apache-2.0）和 `LICENSE_GEMMA.txt`。ControlVLA 部分是 ybao 的 π0.5 近似实现。
