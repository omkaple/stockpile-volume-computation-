"""
3D Visualization and Meshing Module
=====================================
Covers: 3D point cloud visualization, mesh rendering,
multi-view plots, and surface reconstruction methods.

Study Material Topics Covered:
- 3D visualization and meshing
- Surface reconstruction techniques
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    # pyrefly: ignore [missing-import]
    import open3d as o3d
except ImportError:
    print("ERROR: open3d required. pip install open3d")
    sys.exit(1)


# =============================================================================
# 1. Point Cloud 3D Visualization (Matplotlib - saves to file)
# =============================================================================

def plot_point_cloud_3d(pcd, title="3D Point Cloud", output_path=None, subsample=5000):
    """Render a 3D scatter plot of the point cloud and save to file."""
    points = np.asarray(pcd.points)
    
    # Subsample for plotting performance
    if len(points) > subsample:
        idx = np.random.choice(len(points), subsample, replace=False)
        points = points[idx]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=points[:, 2], cmap='viridis', s=1, alpha=0.6
    )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Height (Z)', shrink=0.6)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()
    return output_path


def plot_segmentation_result(ground_pcd, stockpile_pcd, output_path=None, subsample=3000):
    """Visualize ground vs stockpile segmentation in 3D."""
    ground = np.asarray(ground_pcd.points)
    stockpile = np.asarray(stockpile_pcd.points)
    
    if len(ground) > subsample:
        ground = ground[np.random.choice(len(ground), subsample, replace=False)]
    if len(stockpile) > subsample:
        stockpile = stockpile[np.random.choice(len(stockpile), subsample, replace=False)]
    
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(ground[:, 0], ground[:, 1], ground[:, 2],
               c='#95a5a6', s=1, alpha=0.3, label='Ground Plane')
    ax.scatter(stockpile[:, 0], stockpile[:, 1], stockpile[:, 2],
               c='#e74c3c', s=2, alpha=0.7, label='Stockpile')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Ground vs Stockpile Segmentation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, markerscale=8)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()
    return output_path


def plot_mesh_surface(surface, title="Triangle Mesh Surface", output_path=None):
    """Render the reconstructed mesh surface in 3D."""
    vertices = np.asarray(surface.vertices)
    triangles = np.asarray(surface.triangles)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Build polygon collection for triangles (limit for performance)
    max_tri = min(len(triangles), 5000)
    idx = np.random.choice(len(triangles), max_tri, replace=False) if len(triangles) > max_tri else np.arange(len(triangles))
    
    polys = []
    for i in idx:
        tri = triangles[i]
        poly = [vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]]
        polys.append(poly)
    
    mesh_collection = Poly3DCollection(polys, alpha=0.5, edgecolor='#3498db', linewidth=0.2)
    mesh_collection.set_facecolor('#5dade2')
    ax.add_collection3d(mesh_collection)
    
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()
    return output_path


def plot_preprocessing_comparison(pcd_before, pcd_after, output_path=None, subsample=3000):
    """Side-by-side comparison of point cloud before/after preprocessing."""
    pts_before = np.asarray(pcd_before.points)
    pts_after = np.asarray(pcd_after.points)
    
    if len(pts_before) > subsample:
        pts_before = pts_before[np.random.choice(len(pts_before), subsample, replace=False)]
    if len(pts_after) > subsample:
        pts_after = pts_after[np.random.choice(len(pts_after), subsample, replace=False)]
    
    fig = plt.figure(figsize=(18, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts_before[:, 0], pts_before[:, 1], pts_before[:, 2],
                c=pts_before[:, 2], cmap='coolwarm', s=1, alpha=0.5)
    ax1.set_title(f'Before ({len(np.asarray(pcd_before.points)):,} pts)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pts_after[:, 0], pts_after[:, 1], pts_after[:, 2],
                c=pts_after[:, 2], cmap='coolwarm', s=1, alpha=0.5)
    ax2.set_title(f'After ({len(np.asarray(pcd_after.points)):,} pts)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    plt.suptitle('Preprocessing: Before vs After', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()
    return output_path


# =============================================================================
# 2. Surface Reconstruction Methods (Meshing)
# =============================================================================

def poisson_reconstruction(pcd, depth=8):
    """Poisson surface reconstruction (requires normals)."""
    print(f"  Poisson reconstruction (depth={depth})")
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    print(f"  Result: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh, densities


def ball_pivoting_reconstruction(pcd, radii=None):
    """Ball Pivoting Algorithm surface reconstruction (requires normals)."""
    print(f"  Ball Pivoting reconstruction")
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    
    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 1.0, avg_dist * 2.0, avg_dist * 4.0]
    
    radii_vec = o3d.utility.DoubleVector(radii)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vec)
    print(f"  Result: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def alpha_shape_reconstruction(pcd, alpha=0.03):
    """Alpha Shape surface reconstruction."""
    print(f"  Alpha Shape reconstruction (alpha={alpha})")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    print(f"  Result: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


# =============================================================================
# 3. Multi-View Visualization Pipeline
# =============================================================================

def generate_all_visualizations(pcd_raw, ground_pcd, stockpile_pcd, 
                                 stockpile_clean, surface, xy_catalog, 
                                 tri, output_dir):
    """Generate all visualization outputs for the project."""
    from scipy.spatial import Delaunay
    
    os.makedirs(output_dir, exist_ok=True)
    print("\n  === Generating Visualizations ===")
    
    paths = {}
    
    # 1. Raw point cloud
    paths['raw_cloud'] = plot_point_cloud_3d(
        pcd_raw, "Original Point Cloud (Raw)",
        os.path.join(output_dir, "01_raw_point_cloud.png"))
    
    # 2. Segmentation result
    paths['segmentation'] = plot_segmentation_result(
        ground_pcd, stockpile_pcd,
        os.path.join(output_dir, "02_segmentation_result.png"))
    
    # 3. Preprocessed stockpile
    paths['preprocessed'] = plot_point_cloud_3d(
        stockpile_clean, "Preprocessed Stockpile",
        os.path.join(output_dir, "03_preprocessed_stockpile.png"))
    
    # 4. Delaunay triangulation (2D)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.triplot(xy_catalog[:, 0], xy_catalog[:, 1], tri.simplices,
               color='#3498db', linewidth=0.5, alpha=0.7)
    ax.scatter(xy_catalog[:, 0], xy_catalog[:, 1], s=1, color='#e74c3c', alpha=0.5)
    ax.set_title('Delaunay Triangulation (XY Projection)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path_tri = os.path.join(output_dir, "04_delaunay_triangulation.png")
    plt.savefig(path_tri, dpi=150, bbox_inches='tight')
    plt.close()
    paths['triangulation'] = path_tri
    print(f"  Saved: {path_tri}")
    
    # 5. Mesh surface
    paths['mesh'] = plot_mesh_surface(
        surface, "Reconstructed Mesh Surface",
        os.path.join(output_dir, "05_mesh_surface.png"))
    
    # 6. Before vs After
    paths['comparison'] = plot_preprocessing_comparison(
        stockpile_pcd, stockpile_clean,
        os.path.join(output_dir, "06_before_after_comparison.png"))
    
    print(f"\n  All {len(paths)} visualizations saved to: {output_dir}")
    return paths
