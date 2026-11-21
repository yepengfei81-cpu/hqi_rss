# Bimanual pouring: training and rollout code for three π0.5 methods

The data is hqi's 50 bimanual pouring demonstrations (MCAP). In each one, the right arm pours orange juice from a pitcher into a glass cup and then puts the pitcher back. We trained three models on this data with identical hyperparameters:

| Method | Config name |
|---|---|
| baseline | `pi05_hqi_pour50` |
| mask prompt + MCR | `pi05_hqi_pour50_mask` |
| ControlVLA | `pi05_hqi_pour50_controlvla` |

**To run the models on a robot, see [ROLLOUT_HQI.md](ROLLOUT_HQI.md).** It covers where the weights are, the input/output contract, environment setup, the offline self-check, connecting your own robot, and what we learned doing rollouts on AIRBOT.

## What is in this repo

| Path | Contents |
|---|---|
| `rollout_hqi_pour.py` | Robot-agnostic joint-space rollout script: offline self-check, real-robot skeleton (`RealRobot`), session logging |
| `ROLLOUT_HQI.md` | Rollout guide |
| `airbot/hf_mask.py` | Mask generator (Grounding DINO + SAM, slots assigned by box area). Training masks and deployment masks come from this same file |
| `scripts/convert_hqi_mcap.py` | Converts the raw MCAP files to a LeRobot dataset (10 Hz) |
| `scripts/gen_head_masks.py` | Generates the training masks; the command is in the file header |
| `scripts/compute_norm_stats.py`, `scripts/train.py`, `train_hqi_pour.sh` | Normalization statistics, training entry point, cluster sbatch script |
| `src/openpi/` | Model and training code. The three configs are in `src/openpi/training/config.py` (search for `pi05_hqi_pour50`) |
| `assets/` | Normalization statistics for the three configs (`norm_stats.json`; all three are identical, and each checkpoint also has its own copy) |
| `reference/local_inference_eef.py` | Our AIRBOT rollout script, for reading only; the hardware code it depends on is not included |

## Not in this repo (on the cluster)

Cluster directory: `P=/n/netscratch/hankyang_lab/Lab/pengfei`

- Weights: `$P/openpi_hqi_pour/checkpoints/<config>/<exp>/19999`. Deployment needs only `params/` and `assets/`.
- Training data: `$P/pi_data/hqi_pour50` (LeRobot format).
- Training masks: `$P/pi_data/hqi_pour50_masks`.

netscratch purges files after 90 days, so please copy them off soon.

## Origin

The code is based on Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi); see `LICENSE` (Apache-2.0) and `LICENSE_GEMMA.txt`. The ControlVLA part is ybao's approximate reimplementation on π0.5.
