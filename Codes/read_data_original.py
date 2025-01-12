
"""
run with:
cv2 version: 4.10.0
o3d version: 0.18.0
"""
import cv2; print("cv2 version: " + cv2.__version__)
import open3d as o3d; print("o3d version: " + o3d.__version__)
import numpy as np
import os.path
import matplotlib.pyplot as plt
import copy



__OBJECT_NAME__ = "tomatocan"

__PATH__ = "D:/Masters/Project/datas/SampleData/daten/"

__READPATH__ = __PATH__  + __OBJECT_NAME__ + "/"
__MODELPATH__ = "D:/Masters/Project/datas/SampleData/daten/tomatocan/models/model1/tomatocan.stl"

__N_MODELPOINTS_ = 10000    # number of points in pointcstlouds (stl -> pcd)

## tag dimensions (in meter)
__TAG_SIZE__ = 0.064
__TAG_SIZE_OUTER__ = 0.08

##  camera intrinsics
__CAM_FX__ = 570.34222412109381
__CAM_FY__ = 570.3422241210938
__CAM_CX__ = 319.5
__CAM_CY__ = 239.5

## color and depth image parameters
__C_IMG_W__ = 640 # color image, width
__C_IMG_H__ = 480 # color image, height
__D_IMG_W__ = 640 # depth image, width
__D_IMG_H__ = 480 # depth image, height

def load_image(filename):
    img = np.load(__READPATH__ + filename + "_color.npy", allow_pickle=False)
    img = img[:,:,::-1]
    return img

def load_scene_pointcloud(filename):
    depth_data = np.load(__READPATH__ + filename + "_depth.npy", allow_pickle=False)
    depth_arr = np.frombuffer(depth_data, dtype=np.uint16).reshape((__D_IMG_H__, __D_IMG_W__))
    depth_img = o3d.geometry.Image(depth_arr)
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics.set_intrinsics(
        width=__D_IMG_W__,
        height=__D_IMG_H__,
        fx=__CAM_FX__,
        fy=__CAM_FY__,
        cx=__CAM_CX__,
        cy=__CAM_CY__)

    pc = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_img,
        intrinsic=cam_intrinsics,
        extrinsic=np.eye(4)
    )
    return pc

def load_model_pc():
    model_mesh = o3d.io.read_triangle_mesh(__MODELPATH__)
    pc = model_mesh.sample_points_poisson_disk(__N_MODELPOINTS_)
    return pc

if __name__ == '__main__':
    dir = os.listdir(__READPATH__)
    filenames = []
    modelPc = load_model_pc()
    modelPc.paint_uniform_color([1, 0, 0])
    for filename in dir:
        if filename[-4:] == ".txt":
            filenames.append(filename[0:-4])
    # point clouds
    for f in filenames:
        scenePc = load_scene_pointcloud(f)
        trafo_complete = np.array(np.loadtxt(__READPATH__ + f + ".txt"))
        modelPcTafo = copy.copy(modelPc).transform(trafo_complete)

        visu = o3d.visualization.VisualizerWithKeyCallback()
        visu.create_window(window_name="Pointcloud " + f)
        visu.add_geometry(scenePc)
        visu.add_geometry(modelPcTafo)

        visu.run()
        visu.destroy_window()
    # images
    for f in filenames:
        img = load_image(f)
        plt.imshow(img)
        plt.show()
