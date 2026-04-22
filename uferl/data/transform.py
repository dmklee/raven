from typing import List, Tuple
import torch

from uferl.data.types import Obs, Action


class JointTransform:
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(
        self,
        inputs: List[Obs],
        outputs: List[Action]
    ) -> Tuple[List[Obs], List[Obs]]:
        return inputs, outputs
