"""
Volume Computation Methods Module
===================================
Multiple methods for computing stockpile volume from 3D point clouds.

Study Material Topics Covered:
- Volume computation methods (Delaunay, Convex Hull, Voxel-based)
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from functools import reduce

try:
    import open3d as o3d
except ImportError:
    pass


# =============================================================================
# Method 1: Delaunay Triangulation Prism Volume
# =============================================================================

def volume_delaunay_prism(points):
    """Compute volume using Delaunay triangulation + prism summation.
    
    Projects points onto XY plane, creates Delaunay triangulation,
    then sums the volume of each triangular prism down to Z=0.
    
    Best for: Stockpiles scanned from above (2.5D surfaces).
    """
    print("  [Method 1] Delaunay Triangulation Prism")
    
    xy = points[:, :2]
    tri = Delaunay(xy)
    
    volume = 0.0
    for simplex in tri.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        v = abs((z1 + z2 + z3) * (x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3) / 6)
        volume += v
    
    print(f"    Triangles: {len(tri.simplices):,}")
    print(f"    Volume: {volume:.6f} m³")
    return volume, tri


# =============================================================================
# Method 2: Convex Hull Volume
# =============================================================================

def volume_convex_hull(points):
    """Compute volume using 3D Convex Hull.
    
    Creates the smallest convex shape enclosing all points.
    Note: Overestimates for non-convex/irregular stockpiles.
    
    Best for: Quick upper-bound estimate.
    """
    print("  [Method 2] Convex Hull")
    
    hull = ConvexHull(points)
    volume = hull.volume
    
    print(f"    Hull faces: {len(hull.simplices):,}")
    print(f"    Hull area: {hull.area:.6f} m²")
    print(f"    Volume: {volume:.6f} m³")
    return volume, hull


# =============================================================================
# Method 3: Voxel-Based Volume
# =============================================================================

def volume_voxel_based(points, voxel_size=0.01):
    """Compute volume by counting occupied voxels.
    
    Divides space into a 3D grid, counts voxels containing points,
    multiplies by voxel volume. Simple but effective.
    
    Best for: Dense, regular point clouds.
    """
    print(f"  [Method 3] Voxel-Based (size={voxel_size})")
    
    min_bound = points.min(axis=0)
    # Quantize points to voxel grid
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    
    # Count unique occupied voxels
    unique_voxels = set(map(tuple, voxel_indices))
    num_voxels = len(unique_voxels)
    voxel_volume = voxel_size ** 3
    volume = num_voxels * voxel_volume
    
    print(f"    Occupied voxels: {num_voxels:,}")
    print(f"    Voxel volume: {voxel_volume:.8f} m³")
    print(f"    Volume: {volume:.6f} m³")
    return volume, num_voxels


# =============================================================================
# Method 4: Slice/Cross-Section Method
# =============================================================================

def volume_slice_method(points, num_slices=50):
    """Compute volume using horizontal cross-sections.
    
    Slices the point cloud into horizontal layers, computes the
    2D convex hull area of each slice, then integrates.
    
    Best for: Tall, irregular stockpiles.
    """
    print(f"  [Method 4] Slice/Cross-Section (slices={num_slices})")
    
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = z_max - z_min
    slice_thickness = z_range / num_slices
    
    volume = 0.0
    valid_slices = 0
    
    for i in range(num_slices):
        z_lo = z_min + i * slice_thickness
        z_hi = z_lo + slice_thickness
        
        mask = (points[:, 2] >= z_lo) & (points[:, 2] < z_hi)
        slice_pts = points[mask]
        
        if len(slice_pts) >= 3:
            try:
                hull_2d = ConvexHull(slice_pts[:, :2])
                area = hull_2d.volume  # In 2D, ConvexHull.volume = area
                volume += area * slice_thickness
                valid_slices += 1
            except Exception:
                continue
    
    print(f"    Valid slices: {valid_slices}/{num_slices}")
    print(f"    Slice thickness: {slice_thickness:.6f} m")
    print(f"    Volume: {volume:.6f} m³")
    return volume, valid_slices


# =============================================================================
# Compare All Methods
# =============================================================================

def compare_all_methods(points, voxel_size=0.01):
    """Run all volume computation methods and compare results."""
    print("=" * 60)
    print("  VOLUME COMPUTATION - METHOD COMPARISON")
    print("=" * 60)
    print(f"  Input: {len(points):,} points\n")
    
    results = {}
    
    # Method 1: Delaunay
    vol1, _ = volume_delaunay_prism(points)
    results["Delaunay Prism"] = vol1
    print()
    
    # Method 2: Convex Hull
    vol2, _ = volume_convex_hull(points)
    results["Convex Hull"] = vol2
    print()
    
    # Method 3: Voxel
    vol3, _ = volume_voxel_based(points, voxel_size)
    results["Voxel-Based"] = vol3
    print()
    
    # Method 4: Slice
    vol4, _ = volume_slice_method(points)
    results["Slice Method"] = vol4
    
    # Summary table
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<20} {'Volume (m³)':>15}")
    print(f"  {'-'*20} {'-'*15}")
    for method, vol in results.items():
        print(f"  {method:<20} {vol:>15.6f}")
    
    avg = np.mean(list(results.values()))
    std = np.std(list(results.values()))
    print(f"\n  Average: {avg:.6f} m³")
    print(f"  Std Dev: {std:.6f} m³")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Quick test with random data
    np.random.seed(42)
    test_points = np.random.rand(1000, 3) * 0.5
    test_points[:, 2] = test_points[:, 2] * 0.3  # Flatten Z
    compare_all_methods(test_points)
