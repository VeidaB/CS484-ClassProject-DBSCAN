CS484 Class Project: DBSCAN

A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise

This project implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm from scratch, based on the original paper by Ester et al. (1996).

Original Research Paper:
 A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (1996)

Requirements
The following Python libraries are required:
- NumPy
- scikit-learn
- Matplotlib

Install dependencies using:
pip install numpy scikit-learn matplotlib

How to Run:
Run the following command in your terminal:
py dbscan.py

The program will execute DBSCAN on the dataset make_moons and generate a visualization saved as dbscan_result.pdf.

Using a Different Dataset:
To run DBSCAN on a different dataset, modify the Dataset section of dbscan.py.
- Import your desired dataset.
- Assign the dataset to X.
- Adjust eps (neighborhood radius) and min_pts (minimum number of points required to form a core point) as needed.

The algorithm expects X to be a NumPy array with shape (n, d), where each row represents one observation and each column represents a feature.