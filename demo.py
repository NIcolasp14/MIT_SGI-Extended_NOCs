import os
import csv
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import torch
from dataset import NOCSData
import datetime
from scipy import spatial
from scipy.spatial import cKDTree

def create_point_cloud_from_nocs(nocs_map, image, mask=None, scale=1.0):
    print("Initial NOCS map shape:", nocs_map.shape)

    # Convert NOCS coordinates to real-world scale if necessary
    nocs_map_scaled = nocs_map * scale
    print("Scaled NOCS map shape:", nocs_map_scaled.shape)

    if mask is not None:
        nocs_map_scaled = nocs_map_scaled * mask[:, :, np.newaxis]
    
    # Extract valid points based on some criterion (e.g., non-zero depth)
    valid_mask = np.linalg.norm(nocs_map_scaled, axis=2) > 0  # Filter out zero vectors
    print("Valid mask shape:", valid_mask.shape)
    print("Number of true in valid mask:", np.sum(valid_mask))

    # Initialize lists to collect points and colors
    points_3d = []
    colors = []

    # Loop through each point
    for i in range(nocs_map_scaled.shape[0]):
        for j in range(nocs_map_scaled.shape[1]):
            if valid_mask[i, j]:
                point = nocs_map_scaled[i, j]
                color = image[i, j] / 255.0  # Normalize color to [0, 1]
                points_3d.append(point)
                colors.append(color)
                # Debugging: print each point and color
#                print(f"Point: {point}, Color: {color}")
                if np.any(point > 1) or np.any(point < 0):
                    print(f"Invalid point detected: {point}")
                if np.any(color > 1) or np.any(color < 0):
                    print(f"Invalid color detected: {color}")

    points_3d = np.array(points_3d)
    colors = np.array(colors)

    print("Number of points in point cloud:", points_3d.shape[0])
    print("Colors array shape:", colors.shape)

    # Ensure that points and colors arrays have the same length
    assert points_3d.shape[0] == colors.shape[0], "Mismatch between points and colors array lengths."

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
    pcd = pcd.select_by_index(ind)
    print(f"Point cloud size after statistical outlier removal: {len(pcd.points)}")

    # Downsample the point cloud
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # print(f"Point cloud size after downsampling: {len(pcd.points)}")

    return pcd

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

TRAINED_PATH = 'models/NOCS_Trained_2.pth'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Path to specific image
IMAGE_SPECIFIC = None

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6
    OBJ_MODEL_DIR = os.path.join('data', 'obj_models')

config = InferenceConfig()
config.display()

synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

class_map = {
    'bottle': 'bottle',
    'bowl': 'bowl',
    'cup': 'mug',
    'laptop': 'laptop',
}

config.display()

model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)

if config.GPU_COUNT > 0:
    device = torch.device('cuda')
    model.load_state_dict(torch.load(TRAINED_PATH))
else:
    device = torch.device('cpu')
    model.load_state_dict(torch.load(TRAINED_PATH, map_location=torch.device('cpu')))

print("Model to:", device)

model.to(device)

save_dir = 'output_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

now = datetime.datetime.now()

use_camera_data = True
image_id = 1

if use_camera_data:
    camera_dir = os.path.join('data', 'camera')
    dataset = NOCSData(synset_names, 'val')
    dataset.load_camera_scenes(camera_dir)
    dataset.prepare(class_map)

    image = dataset.load_image(image_id)
    depth = dataset.load_depth(image_id)
    image_path = dataset.image_info[image_id]["path"]

    data = "camera/val"
    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
    # intrinsics = np.array([[100.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

else:
    real_dir = os.path.join('data', 'real')
    dataset = NOCSData(synset_names, 'test')
    dataset.load_real_scenes(real_dir)
    dataset.prepare(class_map)

    image = dataset.load_image(image_id)
    depth = dataset.load_depth(image_id)
    image_path = dataset.image_info[image_id]["path"]

    data = "real/test"
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

start_time = datetime.datetime.now()

result = {}
gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
gt_bbox = utils.extract_bboxes(gt_mask)
result['image_id'] = image_id
result['image_path'] = image_path
result['gt_class_ids'] = gt_class_ids
result['gt_bboxes'] = gt_bbox
result['gt_RTs'] = None
result['gt_scales'] = gt_scales

detect = True
if detect:

    if image.shape[2] == 4:
        image = image[:, :, :3]

    with torch.no_grad():
        results = model.detect([image])
        r = results[0]
        rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'], r['coords']

        print(f"rois shape: {rois.shape}")
        print(f"masks shape: {masks.shape}")
        print(f"class_ids shape: {class_ids.shape}")
        print(f"scores shape: {scores.shape}")
        print(f"coords shape: {coords.shape}")

        r['coords'][:, :, :, 2] = 1 - r['coords'][:, :, :, 2]

end_time = datetime.datetime.now()
execution_time = end_time - start_time

print("Time taken for execution:", execution_time)

nocs_map_path = os.path.join(save_dir, 'nocs_map.npy')
np.save(nocs_map_path, coords)
print(f"NOCS map saved to: {nocs_map_path}")

bowl_idx = np.where(class_ids == 2)[0]
if len(bowl_idx) == 0:
    print("No bowl detected.")
else:
    bowl_mask = masks[:, :, bowl_idx[0]].astype(bool)
    bowl_coords = coords[:, :, bowl_idx[0], :]
    bowl_embeddings = results[0]['embeddings'][:, :, bowl_idx[0], :]

    pcd = create_point_cloud_from_nocs(bowl_coords, image, mask=bowl_mask, scale=1.0)
    print("Point cloud generated successfully.")

    o3d.io.write_point_cloud(os.path.join(save_dir, "bowl_pointcloud.ply"), pcd)
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud from NOCS")

def fragmentation_fps(vertices, num_frags):
    """Fragmentation by the furthest point sampling algorithm from the EPOS pipeline."""
    # Start with the origin of the model coordinate system.
    frag_centers = [np.array([0., 0., 0.])]

    # Calculate distances to the center from all the vertices.
    nn_index = spatial.cKDTree(frag_centers)
    nn_dists, _ = nn_index.query(vertices, k=1)

    for _ in range(num_frags):
        # Select the furthest vertex as the next center.
        new_center_ind = np.argmax(nn_dists)
        new_center = vertices[new_center_ind]
        frag_centers.append(vertices[new_center_ind])

        # Update the distances to the nearest center.
        nn_dists[new_center_ind] = -1
        nn_dists = np.minimum(
            nn_dists, np.linalg.norm(vertices - new_center, axis=1))

    # Remove the origin.
    frag_centers.pop(0)
    frag_centers = np.array(frag_centers)

    # Assign vertices to the fragments.
    nn_index = spatial.cKDTree(frag_centers)
    _, vertex_frag_ids = nn_index.query(vertices, k=1)

    return frag_centers, vertex_frag_ids

def visualize_point_cloud(pcd, assignments, frag_centers, num_frags):
    """Visualize the point cloud with distinct colors for each fragment and the seed points."""
    cmap = plt.get_cmap("tab20")  # Use a colormap
    colors = cmap(assignments % cmap.N)[:, :3]  # Map fragment assignments to colors
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a point cloud for the seed points (fragment centers)
    seed_pcd = o3d.geometry.PointCloud()
    seed_pcd.points = o3d.utility.Vector3dVector(frag_centers)
    seed_pcd.paint_uniform_color([1, 0, 0])  # Red color for seed points

    # Combine the fragmented point cloud and seed points for visualization
    o3d.visualization.draw_geometries([pcd, seed_pcd], window_name="Fragmented Point Cloud with Seed Points")


def pool_embeddings(bowl_embeddings, assignments, num_frags):
    height, width, depth = bowl_embeddings.shape
    pooled_features = np.zeros((num_frags, depth))
    frag_counts = np.zeros(num_frags)

    # Debug: Print shapes and some values
    print("Bowl Embeddings Shape:", bowl_embeddings.shape)
    print("Assignments Shape:", assignments.shape)

    for i in range(assignments.shape[0]):
        frag_id = assignments[i]
        if frag_id >= num_frags:
            print(f"Skipping invalid fragment ID {frag_id} at index {i}")
            continue

        # We need to map these back to the 2D image plane of the bowl embeddings
        point = pcd.points[i]  # Assuming the order matches, might need verification
        x = int(point[0] * width)
        y = int(point[1] * height)

        if x >= width or y >= height or x < 0 or y < 0:
            print(f"Skipping invalid coordinates ({x}, {y}) for feature map with shape ({height}, {width})")
            continue

        pooled_features[frag_id] += bowl_embeddings[y, x]
        frag_counts[frag_id] += 1

        # Debug: Print the current pooled feature and count
        print(f"Accumulating fragment {frag_id}: pooled_feature[{frag_id}] = {pooled_features[frag_id]}, count = {frag_counts[frag_id]}")

    # Avoid division by zero
    non_zero_counts = frag_counts > 0
    pooled_features[non_zero_counts] /= frag_counts[non_zero_counts][:, np.newaxis]

    # Debug: Check if pooled_features have been updated
    print("Pooled Features (first 5):", pooled_features[:5])
    print("Fragment Counts (first 5):", frag_counts[:5])
    
    return pooled_features


def save_pooled_features(pooled_features, frag_centers, save_path):
    # Save as NPZ
    np.savez(save_path, pooled_features=pooled_features, frag_centers=frag_centers)
    
    # Save as CSV
    csv_path = save_path + '.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fragment ID', 'Center X', 'Center Y', 'Center Z'] + [f'Feature {i}' for i in range(pooled_features.shape[1])])
        
        for i in range(len(frag_centers)):
            row = [i, frag_centers[i][0], frag_centers[i][1], frag_centers[i][2]] + list(pooled_features[i])
            writer.writerow(row)

    print(f"Saved pooled features to {save_path}.npz and {csv_path}")
    
    
def visualize_frag_centers(frag_centers, save_path):
    seed_pcd = o3d.geometry.PointCloud()
    seed_pcd.points = o3d.utility.Vector3dVector(frag_centers)
    seed_pcd.paint_uniform_color([1, 0, 0])  # Color the seed points red
    
    # Save and visualize the fragment centers
    o3d.io.write_point_cloud(os.path.join(save_path, "frag_centers_pointcloud.ply"), seed_pcd)
    o3d.visualization.draw_geometries([seed_pcd], window_name="Fragment Centers")
    

num_frags = 128  # Number of fragments

# Perform fragmentation on the sampled points
vertices = np.asarray(pcd.points)  # Extract vertices from point cloud
frag_centers, assignments = fragmentation_fps(vertices, num_frags)

# Pool the embeddings and save
pooled_features = pool_embeddings(bowl_embeddings, assignments, num_frags)
save_pooled_features(pooled_features, frag_centers, "output_images/pooled_features")

# Visualize the fragmented point cloud and the seed points
visualize_point_cloud(pcd, assignments, frag_centers, num_frags)
visualize_frag_centers(frag_centers, save_dir)
