"""
Point Cloud Preprocessing Techniques Module
=============================================
Covers: Normal estimation, multiple filtering methods,
format conversion, and point cloud analysis/statistics.

Study Material Topics Covered:
- Point cloud preprocessing techniques
- Point cloud data formats and libraries
- 3D Point Cloud concepts
"""

import os
import sys
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    import open3d as o3d
except ImportError:
    print("ERROR: open3d required. pip install open3d")
    sys.exit(1)


# =============================================================================
# 1. Point Cloud Data Formats & Loading
# =============================================================================

SUPPORTED_FORMATS = {
    ".ply": "Polygon File Format - stores vertices, faces, colors",
    ".pcd": "Point Cloud Data - native PCL/Open3D format",
    ".xyz": "Simple ASCII - X Y Z per line",
    ".xyzn": "ASCII with normals - X Y Z NX NY NZ",
    ".xyzrgb": "ASCII with color - X Y Z R G B",
    ".pts": "Leica scanner format",
    ".obj": "Wavefront OBJ - mesh + point data",
    ".stl": "Stereolithography - triangulated surface",
    ".off": "Object File Format - vertices + faces",
}


def load_any_format(filepath):
    """Load point cloud from any supported format.
    
    Open3D natively supports: PLY, PCD, XYZ, XYZN, XYZRGB, PTS, OBJ, STL, OFF.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {list(SUPPORTED_FORMATS.keys())}")
    
    print(f"  Loading {ext.upper()} file: {filepath}")
    print(f"  Format: {SUPPORTED_FORMATS[ext]}")
    
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        raise ValueError("Loaded point cloud is empty!")
    
    print(f"  Loaded {len(pcd.points):,} points")
    return pcd


def convert_format(input_path, output_path):
    """Convert between point cloud formats."""
    pcd = o3d.io.read_point_cloud(input_path)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"  Converted: {input_path} -> {output_path}")
    return output_path


# =============================================================================
# 2. Point Cloud Statistics & Concepts
# =============================================================================

def compute_statistics(pcd):
    """Compute comprehensive point cloud statistics.
    
    Demonstrates 3D Point Cloud concepts:
    - Bounding box (axis-aligned and oriented)
    - Centroid computation
    - Point density estimation
    - Spatial extent and dimensions
    """
    points = np.asarray(pcd.points)
    
    # Bounding boxes
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    
    # Basic stats
    centroid = pcd.get_center()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    dimensions = max_bound - min_bound
    
    # Point density
    aabb_volume = np.prod(dimensions) if np.all(dimensions > 0) else 0
    density = len(points) / aabb_volume if aabb_volume > 0 else 0
    
    # Distance stats
    distances = np.linalg.norm(points - centroid, axis=1)
    
    stats = {
        "num_points": len(points),
        "centroid": centroid.tolist(),
        "min_bound": min_bound.tolist(),
        "max_bound": max_bound.tolist(),
        "dimensions_xyz": dimensions.tolist(),
        "bounding_box_volume": round(aabb_volume, 6),
        "point_density_per_m3": round(density, 2),
        "has_colors": pcd.has_colors(),
        "has_normals": pcd.has_normals(),
        "mean_distance_from_centroid": round(float(np.mean(distances)), 6),
        "std_distance_from_centroid": round(float(np.std(distances)), 6),
    }
    
    print("\n  === Point Cloud Statistics ===")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    return stats


# =============================================================================
# 3. Preprocessing Techniques
# =============================================================================

def estimate_normals(pcd, radius=0.1, max_nn=30):
    """Estimate point normals using local neighborhood.
    
    Normals are essential for:
    - Surface reconstruction (Poisson, Ball Pivoting)
    - Lighting in 3D visualization
    - Feature extraction
    """
    print(f"  Estimating normals (radius={radius}, max_nn={max_nn})")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    print(f"  Normals estimated for {len(pcd.normals)} points")
    return pcd


def statistical_outlier_removal(pcd, nb_neighbors=30, std_ratio=2.0):
    """Remove statistical outliers based on mean distance to neighbors."""
    print(f"  Statistical outlier removal (neighbors={nb_neighbors}, std={std_ratio})")
    before = len(pcd.points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  Removed {before - len(pcd_clean.points):,} outliers -> {len(pcd_clean.points):,} remain")
    return pcd_clean


def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
    """Remove points with fewer than nb_points neighbors within radius."""
    print(f"  Radius outlier removal (points={nb_points}, radius={radius})")
    before = len(pcd.points)
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_clean = pcd.select_by_index(ind)
    print(f"  Removed {before - len(pcd_clean.points):,} outliers -> {len(pcd_clean.points):,} remain")
    return pcd_clean


def voxel_downsample(pcd, voxel_size=0.05):
    """Downsample using voxel grid filter."""
    print(f"  Voxel downsampling (size={voxel_size})")
    before = len(pcd.points)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  {before:,} -> {len(down.points):,} points")
    return down


def uniform_downsample(pcd, every_k=5):
    """Keep every k-th point."""
    print(f"  Uniform downsampling (every_k={every_k})")
    before = len(pcd.points)
    down = pcd.uniform_down_sample(every_k_points=every_k)
    print(f"  {before:,} -> {len(down.points):,} points")
    return down


def crop_point_cloud(pcd, min_bound, max_bound):
    """Crop point cloud to axis-aligned bounding box."""
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(min_bound),
        max_bound=np.array(max_bound)
    )
    cropped = pcd.crop(bbox)
    print(f"  Cropped: {len(pcd.points):,} -> {len(cropped.points):,} points")
    return cropped


def passthrough_filter(pcd, axis='z', min_val=-np.inf, max_val=np.inf):
    """Filter points along a specific axis (pass-through filter)."""
    points = np.asarray(pcd.points)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    mask = (points[:, axis_idx] >= min_val) & (points[:, axis_idx] <= max_val)
    
    filtered = pcd.select_by_index(np.where(mask)[0])
    print(f"  Passthrough {axis}: {len(pcd.points):,} -> {len(filtered.points):,} points")
    return filtered


# =============================================================================
# 4. Full Preprocessing Pipeline
# =============================================================================

def full_preprocessing_pipeline(filepath, voxel_size=0.05):
    """Run the complete preprocessing pipeline demonstrating all techniques.
    
    Steps:
    1. Load from any supported format
    2. Compute statistics
    3. Estimate normals
    4. Remove outliers (statistical + radius)
    5. Voxel downsample
    6. Final statistics
    """
    print("=" * 60)
    print("  POINT CLOUD PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    pcd = load_any_format(filepath)
    
    # Step 2: Statistics
    stats_before = compute_statistics(pcd)
    
    # Step 3: Normals
    pcd = estimate_normals(pcd)
    
    # Step 4: Outlier removal
    pcd = statistical_outlier_removal(pcd)
    
    # Step 5: Downsample
    pcd = voxel_downsample(pcd, voxel_size)
    
    # Step 6: Re-estimate normals after downsampling
    pcd = estimate_normals(pcd)
    
    # Step 7: Final statistics
    stats_after = compute_statistics(pcd)
    
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return pcd, stats_before, stats_after


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Point Cloud Preprocessing")
    parser.add_argument("--input", "-i", default=os.path.join("src", "data", "stockpile.ply"))
    parser.add_argument("--voxel-size", type=float, default=0.05)
    args = parser.parse_args()
    
    full_preprocessing_pipeline(args.input, args.voxel_size)
