
import open3d as o3d
import numpy as np

def get_scene_pointcloud(depth_img, d_width, d_height, cam_params):

    (fx, fy, cx, cy) = cam_params

    depth_img = o3d.geometry.Image(depth_img)

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=d_width,
        height=d_height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy)

    pc = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_img,
        intrinsic=cam_intrinsics,
        extrinsic=np.eye(4))

    return pc