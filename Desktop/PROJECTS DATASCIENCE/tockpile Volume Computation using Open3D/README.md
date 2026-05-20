# 🏗️ Stockpile Volume Computation using Open3D

A Python-based 3D Computer Vision project that computes the volume of a stockpile (such as sand, gravel, or soil) from a `.ply` point cloud file using Open3D, NumPy, and SciPy.

---

## 📚 Study Material Topics Covered (3D Bharat Task 2)

| # | Topic | Module | Status |
|---|-------|--------|--------|
| 1 | **3D Point Cloud Concepts** | `preprocessing.py` — statistics, bounding box, centroid, density | ✅ |
| 2 | **Point Cloud Preprocessing** | `preprocessing.py` — normal estimation, outlier removal (statistical + radius), voxel/uniform downsampling, cropping, passthrough filter | ✅ |
| 3 | **3D Visualization & Meshing** | `visualization.py` — 3D scatter plots, segmentation views, mesh rendering, before/after comparison, Poisson/Ball Pivoting/Alpha Shape reconstruction | ✅ |
| 4 | **Volume Computation Methods** | `volume_methods.py` — Delaunay prism, Convex Hull, Voxel-based, Slice/Cross-section + comparison | ✅ |
| 5 | **Point Cloud Data Formats & Libraries** | `preprocessing.py` — PLY, PCD, XYZ, XYZN, XYZRGB, PTS, OBJ, STL, OFF support via Open3D | ✅ |

---

## 🚀 Features

✅ Load and process 3D point cloud data (9 formats supported)
✅ Compute point cloud statistics (centroid, bounding box, density, normals)
✅ Detect and segment the ground plane using RANSAC
✅ Separate stockpile points from ground points
✅ Align the stockpile to a reference coordinate system
✅ Remove noise via statistical & radius outlier removal
✅ Downsample dense point clouds (voxel + uniform)
✅ Estimate surface normals
✅ Generate triangulated mesh surfaces (Delaunay, Poisson, Ball Pivoting, Alpha Shape)
✅ Compute volume using 4 different methods with comparison
✅ Generate 6 publication-quality 3D visualizations

---

## 📂 Project Structure

```bash
stockpile-volume-computation/
│
├── stockpile_volume.py       # Main pipeline (entry point)
├── preprocessing.py          # Point cloud preprocessing & data formats
├── visualization.py          # 3D visualization & meshing methods
├── volume_methods.py         # Multiple volume computation methods
│
├── src/
│   ├── data/
│   │   └── stockpile.ply     # Input point cloud data
│   └── stockpile_volume.ipynb # Jupyter notebook version
│
├── output/                   # Generated outputs
│   ├── 01_raw_point_cloud.png
│   ├── 02_segmentation_result.png
│   ├── 03_preprocessed_stockpile.png
│   ├── 04_delaunay_triangulation.png
│   ├── 05_mesh_surface.png
│   ├── 06_before_after_comparison.png
│   ├── stockpile_mesh.ply
│   └── results.json
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* **Python** — Core language
* **Open3D** — Point cloud I/O, processing, surface reconstruction
* **NumPy** — Array operations and math
* **SciPy** — Delaunay triangulation, Convex Hull
* **Matplotlib** — 3D visualization and plotting

---

## 🧠 Pipeline Steps

| Step | Process | Study Topic |
|------|---------|-------------|
| 1 | Load Point Cloud (multi-format) | Data Formats |
| 2 | Compute Statistics (centroid, bounds, density) | 3D Concepts |
| 3 | Segment Ground Plane (RANSAC) | Preprocessing |
| 4 | Separate Ground vs Stockpile | Preprocessing |
| 5 | Align to Axes (Z=0 ground) | Preprocessing |
| 6 | Remove Outliers (statistical) | Preprocessing |
| 7 | Voxel Downsampling | Preprocessing |
| 8 | Estimate Normals | Preprocessing |
| 9 | Delaunay Triangulation | Meshing |
| 10 | Build Triangle Mesh | Meshing |
| 11 | Compute Volume (4 methods) | Volume Methods |
| 12 | Generate 3D Visualizations | Visualization |

---

## 📊 Volume Computation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Delaunay Prism** | Triangulate XY, sum prism volumes to Z=0 | 2.5D surfaces (primary) |
| **Convex Hull** | Smallest convex enclosure volume | Quick upper-bound |
| **Voxel-Based** | Count occupied voxels × voxel volume | Dense regular clouds |
| **Slice Method** | Horizontal cross-sections + area integration | Tall irregular piles |

---

## 🛠️ Installation

```bash
git clone https://github.com/omkaple/stockpile-volume-computation-.git
cd stockpile-volume-computation-
pip install -r requirements.txt
```

---

## ▶️ Run the Project

**Full pipeline (CLI):**
```bash
python stockpile_volume.py
python stockpile_volume.py --input src/data/stockpile.ply --output output/
python stockpile_volume.py --voxel-size 0.03
```

**Preprocessing only:**
```bash
python preprocessing.py --input src/data/stockpile.ply
```

**Volume methods comparison:**
```bash
python volume_methods.py
```

**Jupyter Notebook:**
```bash
jupyter notebook src/stockpile_volume.ipynb
```

---

## 📊 Sample Output

```
============================================================
  STOCKPILE VOLUME COMPUTATION
============================================================

  Loaded 72,198 points from stockpile.ply
  Point density: 484,124.89 pts/m³
  Ground plane: -0.1219x + -0.0428y + 0.9916z + 0.8910 = 0
  Stockpile points: 14,980 (after segmentation)
  After preprocessing: 125 points (outlier removal + downsampling)

  VOLUME COMPARISON:
  Method                   Volume (m³)
  -------------------- ---------------
  Delaunay Prism              0.009911
  Convex Hull                 0.008957
  Voxel-Based                 0.000125
  Slice Method                0.001223

  RESULT: Stockpile Volume = 0.0099 m³
============================================================
```

---

## 📌 Applications

* Mining Industry — Stockpile inventory
* Construction — Material quantity estimation
* Civil Engineering — Earthwork volume
* Terrain Mapping — Topographic analysis
* Drone Surveying — Aerial stockpile measurement

---

## 📈 Future Improvements

* LAS/LAZ format support
* Real-time visualization dashboard
* GPU acceleration
* Automatic stockpile boundary detection
* Web deployment using Streamlit

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Om Kaple**
Aspiring Data Analyst & Machine Learning Enthusiast
