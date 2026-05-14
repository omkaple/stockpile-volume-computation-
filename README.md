# 🏗️ Stockpile Volume Computation using Open3D

A Python-based 3D Computer Vision project that computes the volume of a stockpile (such as sand, gravel, or soil) from a `.ply` point cloud file using Open3D, NumPy, and SciPy.

This project demonstrates practical applications of:

* 3D Point Cloud Processing
* Plane Segmentation using RANSAC
* Mesh Generation
* Delaunay Triangulation
* Volume Estimation from 3D Data

---

# 🚀 Features

✅ Load and visualize 3D point cloud data
✅ Detect and segment the ground plane using RANSAC
✅ Separate stockpile points from ground points
✅ Align the stockpile to a reference coordinate system
✅ Remove noise and outliers from the point cloud
✅ Downsample dense point clouds for optimization
✅ Generate triangulated mesh surfaces
✅ Compute accurate stockpile volume

---

# 📂 Project Structure

```bash
stockpile-volume-computation/
│
├── src/
│   └── stockpile_volume.ipynb
│
├── data/
│   └── stockpile.ply
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

* Python
* Open3D
* NumPy
* SciPy
* Matplotlib
* Jupyter Notebook

---

# 🧠 Workflow Pipeline

| Step | Process                              |
| ---- | ------------------------------------ |
| 1    | Load Point Cloud                     |
| 2    | Segment Ground Plane using RANSAC    |
| 3    | Separate Ground and Stockpile Points |
| 4    | Align Point Cloud to Axes            |
| 5    | Remove Statistical Outliers          |
| 6    | Apply Voxel Downsampling             |
| 7    | Perform Delaunay Triangulation       |
| 8    | Generate Triangle Mesh               |
| 9    | Compute Volume from Mesh             |
| 10   | Display Final Volume Result          |

---

# 📊 Output

Example Output:

```bash
The volume of the stockpile is: 0.0099 m³
```

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/stockpile-volume-computation.git
cd stockpile-volume-computation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install open3d numpy scipy matplotlib
```

---

# ▶️ Run the Project

```bash
jupyter notebook src/stockpile_volume.ipynb
```

Run all notebook cells sequentially to process the point cloud and compute stockpile volume.

---

# 📌 Applications

* Mining Industry
* Construction Site Analysis
* Civil Engineering
* Terrain Mapping
* Material Quantity Estimation
* Drone-based Surveying

---

# 📈 Future Improvements

* Add support for LAS/LAZ files
* Real-time visualization dashboard
* GPU acceleration
* Automatic stockpile boundary detection
* Web deployment using Streamlit

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository and submit pull requests for improvements.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Om Kaple

Aspiring Data Analyst & Machine Learning Enthusiast
