# 📋 Task List — Your Active Projects

You have **two projects** currently. Here's a breakdown of each:

---

## 🏗️ Project 1: Stockpile Volume Computation
**Location:** `c:\Users\VICTUS\Downloads\stockpile-volume-computation-main`
**Type:** Jupyter Notebook (Python + Open3D)
**Status:** ✅ Complete (notebook already has code and output)

### What It Does
Computes the volume of a stockpile (e.g., a pile of gravel/sand) from a **3D point cloud** (`.ply` file) using the **Open3D** library.

### Pipeline Steps (all in `stockpile_volume.ipynb`)

| Step | Task | Description |
|------|------|-------------|
| 1 | **Load Point Cloud** | Read `stockpile.ply` using Open3D |
| 2 | **Segment Ground Plane** | Use RANSAC to find the flat ground plane |
| 3 | **Separate Stockpile** | Split point cloud into ground vs stockpile points |
| 4 | **Align to Axes** | Translate + rotate so ground plane = Z=0 |
| 5 | **Remove Outliers** | Statistical outlier removal (30 neighbors, 2.0 std) |
| 6 | **Downsample** | Voxel downsampling (0.05 voxel size) |
| 7 | **Delaunay Triangulation** | Create 2D triangulation from XY projection |
| 8 | **Build Mesh Surface** | Create 3D triangle mesh from triangulated points |
| 9 | **Compute Volume** | Sum volume under each triangle prism |
| 10 | **Print Result** | Output: `The volume of the stockpile is: 0.0099 m³` |

### Dependencies
- `open3d`, `numpy`, `scipy`, `matplotlib`

### To Run
```bash
# Install dependencies
pip install open3d numpy scipy matplotlib

# Run the notebook
jupyter notebook src/stockpile_volume.ipynb
```

---

