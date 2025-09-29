
import cv2
import numpy as np
import torch
from rich.console import Console
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from nerfstudio.cameras.cameras import Cameras

CONSOLE = Console(width=120)


def interpolate_trajectory(cameras: Cameras, num_views: int = 300):
    """calculate interpolate path"""

    c2ws = np.stack(cameras.camera_to_worlds.cpu().numpy())

    key_rots = Rotation.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    
    if num_views == -1:
        render_c2ws = torch.from_numpy(c2ws)
    else:
        render_c2ws = []
        for i in range(num_views):
            time = float(i) / num_views * (len(c2ws) - 1)
            cam_location = interp(time)
            cam_rot = slerp(time).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = cam_rot
            c2w[:3, 3] = cam_location
            render_c2ws.append(c2w)
        render_c2ws = torch.from_numpy(np.stack(render_c2ws, axis=0))

    # use intrinsic of first camera
    camera_path = Cameras(
        fx=cameras[0].fx,
        fy=cameras[0].fy,
        cx=cameras[0].cx,
        cy=cameras[0].cy,
        height=cameras[0].height,
        width=cameras[0].width,
        camera_to_worlds=render_c2ws[:, :3, :4],
        camera_type=cameras[0].camera_type,
    )
    return camera_path
