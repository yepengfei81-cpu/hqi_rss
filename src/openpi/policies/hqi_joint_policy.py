"""hqi 双臂关节空间数据(倒水, scripts/convert_hqi_mcap.py 转出的 hqi_pour50)的输入/输出变换。

state / actions = 14 维绝对关节角: [左 j1..j6, 左夹爪, 右 j1..j6, 右夹爪], 夹爪 0..1。
三路相机: cam_high(俯视) -> base_0_rgb, cam_left_wrist -> left_wrist_0_rgb, cam_right_wrist -> right_wrist_0_rgb, 全部有效。
mask prompt / ControlVLA 的 mask 只给俯视相机(与 AIRBOT 三个任务只给头部相机一致), 手腕相机由模型补零。
不用我们 AIRBOT 管线里的夹爪点云 / 区域 token / W' 末端坐标系(那些依赖 AIRBOT 正运动学和我们自己的夹爪模型)。
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class HqiJointInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/cam_high"]),
                "left_wrist_0_rgb": _parse_image(data["observation/cam_left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/cam_right_wrist"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if data.get("observation/base_region_mask") is not None:
            inputs["region_mask"] = {
                "base_0_rgb": (np.asarray(data["observation/base_region_mask"]) > 0).astype(np.float32)}
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class HqiJointOutputs(transforms.DataTransformFn):
    """模型输出 32 维里取前 14 维(已由 AbsoluteActions 还原成绝对关节角)。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
