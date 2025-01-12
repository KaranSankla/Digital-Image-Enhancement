import cv2; print("cv2 version: " + cv2.__version__)
import open3d as o3d; print("o3d version: " + o3d.__version__)
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
__READPATH__  = r"D:\Masters\Project\datas\SampleData\daten\tomatocan"

# Model path
__MODELPATH__ = r"D:\Masters\Project\datas\SampleData\daten\models\model1\tomatocan.stl"

__N_MODELPOINTS_ = 10000    # number of points in pointcstlouds (stl -> pcd)
path = r"D:\Masters\Project\datas\SampleData\daten\models\model1\tomatocan.stl"
mesh = o3d.io.read_triangle_mesh(path)

if mesh.is_empty():
    print("Mesh is empty or invalid!")
else:
    print("Mesh loaded successfully.")
    o3d.visualization.draw_geometries([mesh])


# def load_model_pc():
#     print(f"Loading model from: {__MODELPATH__}")
#     model_mesh = o3d.io.read_triangle_mesh(__MODELPATH__)
#     if model_mesh.is_empty():
#         raise RuntimeError("Model mesh is empty or invalid.")
#     print("Model mesh loaded.")
#
#     pc = model_mesh.sample_points_poisson_disk(__N_MODELPOINTS_)
#     if pc.is_empty():
#         raise RuntimeError("Point cloud sampling failed. Result is empty.")
#     print("Point cloud sampled successfully.")
#     return pc
# modelPc = load_model_pc()
# print("Model point cloud loaded successfully.")


# def load_scene_pointcloud(filename):
#     path_1 = os.path.join(__READPATH__, f"{filename}.txt")  # Assuming depth file for scene
#     print(f"Loading scene point cloud from: {path_1}")
#     try:
#         data = np.load(path_1)
#         print(f"Scene data shape: {data.shape}")
#
#         # Ensure the data is of the correct type and shape (n, 3)
#         if data.ndim != 2 or data.shape[1] != 3:
#             raise ValueError(f"Invalid data shape: {data.shape}. Expected shape (n, 3).")
#
#         # Convert the data to the correct type
#         pc = o3d.geometry.PointCloud()
#         pc.points = o3d.utility.Vector3dVector(data.astype(np.float64))  # Convert to float64 if necessary
#
#         return pc
#     except Exception as e:
#         print(f"Failed to load scene point cloud: {e}")
#         return o3d.geometry.PointCloud()


if __name__ == '__main__':
    # Rename 'dir' to 'directory_files' to avoid the conflict with the built-in 'dir' function
    directory_files = os.listdir(path='D:\\Masters\\Project\\datas\\SampleData\\daten\\tomatocan')
    print(f"Files in directory: {directory_files}")
    filenames = []

    # Load model point cloud

    for filename in directory_files:
        if filename.endswith(".txt"):
            filenames.append(filename[:-4])  # Remove the ".txt" extension
    for filename in filenames:
        print(f"Processing: {filename}")

        # Load scene point cloud
        # scenePc = load_scene_pointcloud(filename)
        # if scenePc.is_empty():
        #     print(f"Scene point cloud for {filename} is empty.")
        #     continue


    # Load transformation matrix
        trafo_path = os.path.join('D:\\Masters\\Project\\datas\\SampleData\\daten\\tomatocan', f"{filename}.txt")
        if not os.path.exists(trafo_path):
            print(f"Transformation file missing: {trafo_path}")
        continue

        trafo_complete = np.loadtxt(trafo_path)
        if trafo_complete.shape != (4, 4):
            print(f"Invalid transformation matrix shape for {filename}: {trafo_complete.shape}")
        continue



    # Apply transformation
    #modelPcTafo = copy.copy(modelPc).transform(trafo_complete)

    # Visualize
    #visu = o3d.visualization.VisualizerWithKeyCallback()
    #visu.create_window(window_name=f"Pointcloud {filename}")
    #visu.add_geometry(scenePc)
    #visu.add_geometry(modelPcTafo)
    visu.run()
    visu.destroy_window()