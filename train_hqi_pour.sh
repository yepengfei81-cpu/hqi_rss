#!/bin/bash
#SBATCH --job-name=hqi_pour
#SBATCH --account=hankyang_lab
#SBATCH --partition=gpu,seas_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%j.out
# hqi 双臂倒水数据(50 条)上的三个方法: baseline / mask(hqi 的 mask 提示 + MCR) / ControlVLA, 在 hqi 账号 / hankyang_lab 下跑:
#   cd $P/openpi_hqi_pour
#   sbatch -J pour_base train_hqi_pour.sh pi05_hqi_pour50            pour50_base_v1
#   sbatch -J pour_mask train_hqi_pour.sh pi05_hqi_pour50_mask       pour50_mask_v1
#   sbatch -J pour_cvla train_hqi_pour.sh pi05_hqi_pour50_controlvla pour50_controlvla_v1
# 数据 $P/pi_data/hqi_pour50 + hqi_pour50_masks, 2 卡、全局 batch 48、20k 步, 三个配置 norm_stats 同一份。
# 代码在 $P/openpi_hqi_pour(独立目录); Python 环境借用 $P/openpi/.venv, 靠 PYTHONPATH 让 import openpi 解析到本目录的 src。
# 超时就再 sbatch 一次, 会从最新 checkpoint 续训(只认纯数字步数目录)。
P=/n/netscratch/hankyang_lab/Lab/pengfei
C=$P/openpi_hqi_pour
export HF_LEROBOT_HOME=$P/pi_data OPENPI_DATA_HOME=$P/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# torchcodec 解码视频要 FFmpeg 4-7 的动态库, 集群 module 只有 0.8 的老版本; 用 mamba 装在 $P/ffmpeg_env 的 FFmpeg 6
export LD_LIBRARY_PATH=$P/ffmpeg_env/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTHONPATH=$C/src:$C/packages/openpi-client/src
cd $C
CONFIG=${1:?用法: sbatch -J <名字> train_controlvla_hqi.sh <config> <exp>}
EXP=${2:?缺 exp 名}
case "$CONFIG" in pi05_hqi_pour50|pi05_hqi_pour50_mask|pi05_hqi_pour50_controlvla) ;; *) echo "这个脚本只跑 pi05_hqi_pour50[_mask|_controlvla], 收到 $CONFIG"; exit 2;; esac
CKPT_DIR="checkpoints/$CONFIG/$EXP"
if [ -d "$CKPT_DIR" ] && ls "$CKPT_DIR" 2>/dev/null | grep -qE '^[0-9]+$'; then
    MODE=--resume
else
    MODE=--overwrite
fi
# wandb 同 baseline/mask: 只在这个任务里设 WANDB_API_KEY, 不动 hqi 的登录; 没有 key 文件就不开。
if [ -s $P/.wandb_key ] && { [ "$MODE" = --overwrite ] || [ -f "$CKPT_DIR/wandb_id.txt" ]; }; then
    export WANDB_API_KEY=$(cat $P/.wandb_key) WANDB_DIR=$P WANDB_CONFIG_DIR=$P/wandb_config WANDB_CACHE_DIR=$P/wandb_cache
    WANDB_FLAG=--wandb-enabled
else
    WANDB_FLAG=--no-wandb-enabled
fi
echo ">>> $(date) $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $MODE"
set -e
# 开训前确认 import 到的是本目录的代码, 否则直接退出(防止误跑成 $P/openpi 里的 baseline 代码)
$P/openpi/.venv/bin/python -c "import openpi, sys; f = openpi.__file__; print('>>> openpi from', f); sys.exit(0 if f.startswith('$C/') else 3)"
# 直接用 .venv 的 python, 不经过 uv: uv 在 netscratch(NFS) 上拿缓存锁会一直卡住。
# keep-period=None + orbax max_to_keep=1: 只留最新一档
$P/openpi/.venv/bin/python scripts/train.py $CONFIG \
  --exp-name=$EXP $MODE \
  --save-interval=2000 \
  --keep-period=None \
  $WANDB_FLAG
