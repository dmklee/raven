from typing import Tuple, Optional
from dataclasses import dataclass
from torch import Tensor


class Obs:
    pass


class Sensor:
    '''Sensors produce observations'''
    obs_horizon: int
    obs_shape: tuple
    info: dict
    def get_obs_info(self) -> Obs:
        pass


class RGBCamera(Sensor):
    def __init__(
        self,
        img_shape: Tuple[int, int],
        intrinsics: Tensor,
        dist_coeffs: Optional[Tensor]=None,
        extrinsics: Optional[Tensor]=None
    ):
        self.img_shape = img_shape
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.extrinsics = None


class Actuator:
    '''Actuators produce actions'''
    pass

class Action:
    pass

# class GripperAction(Action):
    # def __init__(self):
        # pass
