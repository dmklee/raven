from typing import Optional, Tuple

import torch
from torch import nn, Tensor

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_rays(rays, origins=None, colors=None):
    if origins is None:
        origins = np.zeros((len(rays), 3))

    if colors is None:
        colors = len(rays) * ['b']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(rays)):
        ax.quiver(
            origins[i, 0], origins[i, 1], origins[i, 2],
            rays[i, 0], rays[i, 1], rays[i, 2],
            color=colors[i], alpha=0.8
        )

    ax.set_xlabel('X-axis', labelpad=15)
    ax.set_ylabel('Y-axis', labelpad=15)
    ax.set_zlabel('Z-axis', labelpad=15)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()


def get_opencv_rays(h: int, w: int, align_corners=False) -> Tensor:
    '''
          Z
         /
        ----X
        |
        Y
    '''
    theta = torch.eye(3)[:2].unsqueeze(0)
    uv = nn.functional.affine_grid(theta, (1, 1, h, w), align_corners=align_corners)
    uv = uv.flatten(0, 2)
    z = torch.ones(uv.shape[0], 1)
    uvz = torch.cat([uv, z], dim=1)
    return uvz # shape: (h*w, 3)



if __name__ == "__main__":
    H, W = 20, 20
    rays = get_opengl_rays(H, W)

    visualize_rays(rays)
