import numpy as np
import math

from scene.cameras import Camera
from scipy.spatial.transform import Rotation


def generate_new_cam(r_d, tx, resolution=180):

    
    # rot = Rotation.from_rotvec(r_d).as_matrix()
    rot = r_d
    
    trans = tx


    fovx = np.deg2rad(90)
    fovy = np.deg2rad(90)

    cam = Camera(R=rot,colmap_id=None, T=trans, FoVx=fovx, FoVy=fovy, image=None, image_name=None, uid=None,invdepthmap=None,depth_params=None )
    cam.image_width=resolution*2
    cam.image_height=90
    
    return cam


