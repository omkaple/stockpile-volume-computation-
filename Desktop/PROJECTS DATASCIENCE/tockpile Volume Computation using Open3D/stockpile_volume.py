"""
Stockpile Volume Computation
=============================
Computes the volume of a stockpile from a 3D point cloud using Open3D.

Study Material Topics Covered:
- 3D Point Cloud concepts (statistics, bounding box, normals)
- Point cloud preprocessing techniques (filtering, downsampling, normals)
- 3D visualization and meshing (3D plots, surface reconstruction)
- Volume computation methods (Delaunay, Convex Hull, Voxel, Slice)
- Point cloud data formats and libraries (PLY, PCD, XYZ, Open3D, SciPy)

Pipeline:
    1. Load point cloud from PLY file (multi-format support)
    2. Compute point cloud statistics (3D concepts)
    3. Segment ground plane using RANSAC
    4. Separate stockpile points from ground
    5. Align point cloud (translate + rotate so ground = Z=0)
    6. Remove statistical outliers (preprocessing)
    7. Voxel downsample for efficiency (preprocessing)
    8. Estimate normals (preprocessing)
    9. Delaunay triangulation on XY projection
   10. Build triangle mesh surface (meshing)
   11. Compute volume (multiple methods comparison)
   12. Generate 3D visualizations

Usage:
    python stockpile_volume.py
    python stockpile_volume.py --input path/to/pointcloud.ply
    python stockpile_volume.py --input data.ply --voxel-size 0.03 --output results/
"""

import os
import sys
import math
import argparse
import json
from datetime import datetime
from functools import reduce

import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for deployment
import matplotlib.pyplot as plt

try:
    # pyrefly: ignore [missing-import]
    import open3d as o3d
except ImportError:
    print("ERROR: open3d is required. Install with: pip install open3d")
    print("       Note: open3d requires Python 3.8-3.12")
    sys.exit(1)


# =============================================================================
# Configuration Defaults
# =============================================================================
DEFAULT_PLY_PATH = os.path.join("src", "data", "stockpile.ply")
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_VOXEL_SIZE = 0.05
DEFAULT_RANSAC_DISTANCE_THRESHOLD = 0.01
DEFAULT_RANSAC_N = 3
DEFAULT_RANSAC_ITERATIONS = 10000
DEFAULT_OUTLIER_NEIGHBORS = 30
DEFAULT_OUTLIER_STD_RATIO = 2.0


# =============================================================================
# Core Functions
# =============================================================================

def load_point_cloud(ply_path):
    """Step 1: Load point cloud from PLY file."""
    print(f"  [1/9] Loading point cloud from: {ply_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = len(pcd.points)
    print(f"        Loaded {num_points:,} points")
    
    if num_points == 0:
        raise ValueError("Point cloud is empty!")
    
    return pcd


def segment_ground_plane(pcd, distance_threshold, ransac_n, num_iterations):
    """Step 2: Segment the ground plane using RANSAC."""
    print(f"  [2/9] Segmenting ground plane (RANSAC, threshold={distance_threshold})")
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    [a, b, c, d] = plane_model
    print(f"        Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"        Ground points: {len(inliers):,}")
    
    return plane_model, inliers


def separate_stockpile(pcd, inliers):
    """Step 3: Separate stockpile from ground plane."""
    print(f"  [3/9] Separating stockpile from ground")
    
    plane_pcd = pcd.select_by_index(inliers)
    stockpile_pcd = pcd.select_by_index(inliers, invert=True)
    
    print(f"        Stockpile points: {len(stockpile_pcd.points):,}")
    
    return plane_pcd, stockpile_pcd


def align_to_axes(plane_pcd, stockpile_pcd, plane_model):
    """Step 4: Translate and rotate so ground plane aligns with Z=0."""
    print(f"  [4/9] Aligning point cloud to axes (ground -> Z=0)")
    
    [a, b, c, d] = plane_model
    
    # Translate
    plane_pcd = plane_pcd.translate((0, 0, d / c))
    stockpile_pcd = stockpile_pcd.translate((0, 0, d / c))
    
    # Rotate
    cos_theta = c / math.sqrt(a**2 + b**2 + c**2)
    sin_theta = math.sqrt((a**2 + b**2) / (a**2 + b**2 + c**2))
    u_1 = b / math.sqrt(a**2 + b**2)
    u_2 = -a / math.sqrt(a**2 + b**2)
    
    rotation_matrix = np.array([
        [cos_theta + u_1**2 * (1 - cos_theta), u_1 * u_2 * (1 - cos_theta), u_2 * sin_theta],
        [u_1 * u_2 * (1 - cos_theta), cos_theta + u_2**2 * (1 - cos_theta), -u_1 * sin_theta],
        [-u_2 * sin_theta, u_1 * sin_theta, cos_theta]
    ])
    
    plane_pcd.rotate(rotation_matrix)
    stockpile_pcd.rotate(rotation_matrix)
    
    print(f"        Rotation applied successfully")
    
    return plane_pcd, stockpile_pcd


def remove_outliers(stockpile_pcd, nb_neighbors, std_ratio):
    """Step 5: Remove statistical outliers."""
    print(f"  [5/9] Removing outliers (neighbors={nb_neighbors}, std_ratio={std_ratio})")
    
    points_before = len(stockpile_pcd.points)
    cl, ind = stockpile_pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    stockpile_pcd = stockpile_pcd.select_by_index(ind)
    points_after = len(stockpile_pcd.points)
    removed = points_before - points_after
    
    print(f"        Removed {removed:,} outliers ({points_after:,} points remaining)")
    
    return stockpile_pcd


def voxel_downsample(stockpile_pcd, voxel_size):
    """Step 6: Voxel downsampling for efficiency."""
    print(f"  [6/9] Voxel downsampling (voxel_size={voxel_size})")
    
    points_before = len(stockpile_pcd.points)
    downpcd = stockpile_pcd.voxel_down_sample(voxel_size=voxel_size)
    points_after = len(downpcd.points)
    
    print(f"        {points_before:,} -> {points_after:,} points")
    
    return downpcd


def create_delaunay_triangulation(downpcd):
    """Step 7: Create Delaunay triangulation on XY projection."""
    print(f"  [7/9] Creating Delaunay triangulation")
    
    xyz = np.asarray(downpcd.points)
    xy_catalog = []
    for point in xyz:
        xy_catalog.append([point[0], point[1]])
    
    tri = Delaunay(np.array(xy_catalog))
    
    print(f"        Created {len(tri.simplices):,} triangles")
    
    return xyz, np.array(xy_catalog), tri


def build_mesh_surface(xyz, tri):
    """Step 8: Build triangle mesh surface."""
    print(f"  [8/9] Building triangle mesh surface")
    
    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(xyz)
    surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
    
    print(f"        Mesh: {len(surface.vertices)} vertices, {len(surface.triangles)} triangles")
    
    return surface


def get_triangles_vertices(triangles, vertices):
    """Extract vertex coordinates for each triangle."""
    triangles_vertices = []
    for triangle in triangles:
        new_triangles_vertices = [
            vertices[triangle[0]],
            vertices[triangle[1]],
            vertices[triangle[2]]
        ]
        triangles_vertices.append(new_triangles_vertices)
    return np.array(triangles_vertices)


def volume_under_triangle(triangle):
    """Compute volume of the prism under a single triangle down to Z=0."""
    p1, p2, p3 = triangle
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    return abs(
        (z1 + z2 + z3) * (x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3) / 6
    )


def compute_volume(surface):
    """Step 9: Compute total stockpile volume."""
    print(f"  [9/9] Computing stockpile volume")
    
    triangles = np.asarray(surface.triangles)
    vertices = np.asarray(surface.vertices)
    
    tri_verts = get_triangles_vertices(triangles, vertices)
    volume = reduce(
        lambda a, b: a + volume_under_triangle(b),
        tri_verts,
        0
    )
    
    return volume


# =============================================================================
# Visualization & Output
# =============================================================================

def save_triangulation_plot(xy_catalog, tri, output_dir):
    """Save Delaunay triangulation plot to file."""
    plot_path = os.path.join(output_dir, "delaunay_triangulation.png")
    print(f"\n  Saving triangulation plot -> {plot_path}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.triplot(xy_catalog[:, 0], xy_catalog[:, 1], tri.simplices,
               color='#3498db', linewidth=0.5, alpha=0.7)
    ax.scatter(xy_catalog[:, 0], xy_catalog[:, 1], s=1, color='#e74c3c', alpha=0.5)
    ax.set_title('Delaunay Triangulation (XY Projection)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def save_results(volume, output_dir, ply_path, params):
    """Save computation results to JSON file."""
    results_path = os.path.join(output_dir, "results.json")
    print(f"  Saving results -> {results_path}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_file": os.path.abspath(ply_path),
        "volume_m3": round(volume, 6),
        "volume_m3_display": f"{round(volume, 4)} m3",
        "parameters": {
            "voxel_size": params["voxel_size"],
            "ransac_distance_threshold": params["ransac_distance_threshold"],
            "ransac_n": params["ransac_n"],
            "ransac_iterations": params["ransac_iterations"],
            "outlier_neighbors": params["outlier_neighbors"],
            "outlier_std_ratio": params["outlier_std_ratio"],
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path


def save_processed_mesh(surface, output_dir):
    """Save the processed mesh to PLY file."""
    mesh_path = os.path.join(output_dir, "stockpile_mesh.ply")
    print(f"  Saving processed mesh -> {mesh_path}")
    surface.paint_uniform_color([0.0, 0.5, 1.0])
    o3d.io.write_triangle_mesh(mesh_path, surface)
    return mesh_path


# =============================================================================
# Main Pipeline
# =============================================================================

def compute_stockpile_volume(
    ply_path=DEFAULT_PLY_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    voxel_size=DEFAULT_VOXEL_SIZE,
    ransac_distance_threshold=DEFAULT_RANSAC_DISTANCE_THRESHOLD,
    ransac_n=DEFAULT_RANSAC_N,
    ransac_iterations=DEFAULT_RANSAC_ITERATIONS,
    outlier_neighbors=DEFAULT_OUTLIER_NEIGHBORS,
    outlier_std_ratio=DEFAULT_OUTLIER_STD_RATIO,
):
    """
    Full pipeline to compute stockpile volume from a point cloud.
    
    Returns:
        dict: Results including volume, paths, and parameters.
    """
    print("=" * 60)
    print("  STOCKPILE VOLUME COMPUTATION")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    params = {
        "voxel_size": voxel_size,
        "ransac_distance_threshold": ransac_distance_threshold,
        "ransac_n": ransac_n,
        "ransac_iterations": ransac_iterations,
        "outlier_neighbors": outlier_neighbors,
        "outlier_std_ratio": outlier_std_ratio,
    }
    
    # === PIPELINE ===
    # Step 1: Load
    pcd = load_point_cloud(ply_path)
    pcd_raw = o3d.geometry.PointCloud(pcd)  # Keep copy for visualization
    
    # Step 1b: Point Cloud Statistics (3D Concepts)
    try:
        from preprocessing import compute_statistics, estimate_normals
        print("\n  --- Point Cloud Statistics (3D Concepts) ---")
        compute_statistics(pcd)
    except ImportError:
        print("  [info] preprocessing module not found, skipping statistics")
    
    # Step 2: Segment ground
    plane_model, inliers = segment_ground_plane(
        pcd, ransac_distance_threshold, ransac_n, ransac_iterations
    )
    
    # Step 3: Separate
    plane_pcd, stockpile_pcd = separate_stockpile(pcd, inliers)
    stockpile_raw = o3d.geometry.PointCloud(stockpile_pcd)  # Pre-processing copy
    
    # Step 4: Align
    plane_pcd, stockpile_pcd = align_to_axes(plane_pcd, stockpile_pcd, plane_model)
    
    # Step 5: Remove outliers
    stockpile_pcd = remove_outliers(stockpile_pcd, outlier_neighbors, outlier_std_ratio)
    
    # Step 6: Downsample
    downpcd = voxel_downsample(stockpile_pcd, voxel_size)
    
    # Step 6b: Estimate normals (Preprocessing technique)
    try:
        downpcd = estimate_normals(downpcd)
    except (ImportError, NameError):
        print("  [info] Skipping normal estimation")
    
    # Step 7: Triangulate
    xyz, xy_catalog, tri = create_delaunay_triangulation(downpcd)
    
    # Step 8: Build mesh
    surface = build_mesh_surface(xyz, tri)
    
    # Step 9: Compute volume (primary method)
    volume = compute_volume(surface)
    
    # Step 9b: Compare multiple volume methods
    volume_comparison = None
    try:
        from volume_methods import compare_all_methods
        print()
        volume_comparison = compare_all_methods(xyz)
    except ImportError:
        print("  [info] volume_methods module not found, skipping comparison")
    
    # === OUTPUT ===
    print()
    print("=" * 60)
    print(f"  [OK] RESULT: Stockpile Volume = {round(volume, 4)} m3")
    print("=" * 60)
    print()
    
    # Save outputs
    plot_path = save_triangulation_plot(xy_catalog, tri, output_dir)
    results_path = save_results(volume, output_dir, ply_path, params)
    mesh_path = save_processed_mesh(surface, output_dir)
    
    # Generate all 3D visualizations
    try:
        from visualization import generate_all_visualizations
        viz_paths = generate_all_visualizations(
            pcd_raw, plane_pcd, stockpile_raw,
            downpcd, surface, xy_catalog, tri, output_dir
        )
    except ImportError:
        print("  [info] visualization module not found, skipping 3D plots")
    
    print()
    print("  All outputs saved to:", os.path.abspath(output_dir))
    print("=" * 60)
    
    return {
        "volume_m3": round(volume, 6),
        "volume_comparison": volume_comparison,
        "plot_path": plot_path,
        "results_path": results_path,
        "mesh_path": mesh_path,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute stockpile volume from a 3D point cloud (.ply)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stockpile_volume.py
  python stockpile_volume.py --input path/to/cloud.ply
  python stockpile_volume.py --input data.ply --voxel-size 0.03 --output results/
        """
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_PLY_PATH,
        help=f"Path to input PLY file (default: {DEFAULT_PLY_PATH})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        help=f"Voxel size for downsampling (default: {DEFAULT_VOXEL_SIZE})"
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=DEFAULT_RANSAC_DISTANCE_THRESHOLD,
        help=f"RANSAC distance threshold (default: {DEFAULT_RANSAC_DISTANCE_THRESHOLD})"
    )
    parser.add_argument(
        "--outlier-neighbors",
        type=int,
        default=DEFAULT_OUTLIER_NEIGHBORS,
        help=f"Number of neighbors for outlier removal (default: {DEFAULT_OUTLIER_NEIGHBORS})"
    )
    parser.add_argument(
        "--outlier-std",
        type=float,
        default=DEFAULT_OUTLIER_STD_RATIO,
        help=f"Std ratio for outlier removal (default: {DEFAULT_OUTLIER_STD_RATIO})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    result = compute_stockpile_volume(
        ply_path=args.input,
        output_dir=args.output,
        voxel_size=args.voxel_size,
        ransac_distance_threshold=args.ransac_threshold,
        outlier_neighbors=args.outlier_neighbors,
        outlier_std_ratio=args.outlier_std,
    )
