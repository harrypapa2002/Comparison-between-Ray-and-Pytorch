# Comparing Ray and PyTorch for Scalable Big Data Analytics and Machine Learning

This repository provides implementations of three data-intensive tasks—**Lesion Classification**, **NYC Yellow Taxi Trip Clustering**, and **PageRank**—using both **Ray** and **PyTorch**. The goal is to offer a clear comparison of how each framework handles large-scale data analytics and machine learning workflows. The project is part of the *Analysis and Design of Information Systems* course (9th semester) at the School of Electrical and Computer Engineering at the National Technical University of Athens (NTUA).

A detailed report covering the experimental setup, methodology, and results is included in this repository.

---

**Contributors**  
- **Antonios Alexiadis** – [el20167@mail.ntua.gr](mailto:el20167@mail.ntua.gr)  
- **Charidimos Papadakis** – [el20022@mail.ntua.gr](mailto:el20022@mail.ntua.gr)  
- **Nikolaos Bothos Vouterakos** – [el20158@mail.ntua.gr](mailto:el20158@mail.ntua.gr)  

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [About the Project](#about-the-project)
- [Additional Information](#additional-information)
- [Contributing](#contributing)
- [License](#license)

---

## Repository Structure

The repository is divided into three top-level folders, each corresponding to one of the tasks explored:

<details>
<summary><strong>1. LesionClassification/</strong></summary>

```bash
LesionClassification/
├── data/
├── logs/
├── output/
├── results/
├── compare_results.py
├── launch_namenode.sh
├── lesion_classification_pytorch.py
├── lesion_classification_ray.py
└── preprocessing_functions.py
```

**Key Points:**
- **data/**: Lesion datasets in `.xlsx` format.
- **logs/**: Example log files generated during training or execution.
- **output/**: Figures and CSV files summarizing performance metrics such as execution times and mean evaluation metrics.
- **results/**: JSON output files summarizing experiment metrics.
- **compare_results.py**: Utility script for comparing evaluation metrics and execution times.
- **launch_namenode.sh**: Placeholder script for initializing environment resources.
- **lesion_classification_pytorch.py** / **lesion_classification_ray.py**: Implementations of lesion classification using PyTorch or Ray.
- **preprocessing_functions.py**: Shared data loading and preprocessing methods.
</details>

<details>
<summary><strong>2. NYC_YellowTaxiTrip/</strong></summary>

```bash
NYC_YellowTaxiTrip/
├── maps/
├── output/
├── results/
├── compare_results.py
├── launch_namenode.sh
├── nyc_taxi_pytorch_cluster.py
└── nyc_taxi_ray_cluster.py
```

**Key Points:**
- **maps/**: HTML visualizations of cluster centers.
- **output/**: Figures and CSV files summarizing performance metrics such as execution times and silhouette scores.
- **results/**: JSON results from clustering runs.
- **compare_results.py**: Utility script for comparing clustering outputs and execution times.
- **launch_namenode.sh**: Placeholder script for environment initialization.
- **nyc_taxi_pytorch_cluster.py** / **nyc_taxi_ray_cluster.py**: Clustering implementations using PyTorch or Ray.
</details>

<details>
<summary><strong>3. PageRank/</strong></summary>

```bash
PageRank/
├── output
├── results/
│   └── twitter7/
├── graph_results.py
├── launch.sh
├── pagerank-ray.py
├── pagerank.py
└── unzip-resize.py
```

**Key Points:**
- **output/**: Charts illustrating runtime comparisons for different data sizes and numbers of nodes.
- **results/**: Logs and text files documenting PageRank outputs on subsets of the Twitter7 dataset.
- **graph_results.py**: Script for processing and visualizing PageRank results.
- **launch.sh**: Placeholder script for environment startup.
- **pagerank-ray.py** / **pagerank.py**: Ray and PyTorch versions of the PageRank algorithm.
- **unzip-resize.py**: Utility to extract or resize datasets as needed.
</details>

---

## About the Project

This project explores and compares two distributed computing frameworks—**Ray** and **PyTorch**—in the context of big data analytics and machine learning. The implementations cover the following tasks:

- **Lesion Classification**  
  Evaluate the performance of Ray and PyTorch in running a lesion classification pipeline on medical imaging and tabular data.

- **NYC Yellow Taxi Trip Clustering**  
  Perform geospatial clustering on NYC taxi trip data to uncover common pickup locations and assess clustering quality.

- **PageRank**  
  Compute the PageRank of nodes in a large-scale Twitter follower network (Twitter7 dataset) to identify influential users.

Each task is implemented with both Ray and PyTorch to highlight their respective strengths and trade-offs.

---

## Additional Information

- **Detailed Report**: A comprehensive report describing the setup, methodology, experimental results, and insights is available in the repository.
- **Execution Instructions**: The report also contains detailed steps on how to run the code for each framework and each task.
- **Environment**: The code is designed for distributed execution in a cluster environment (e.g., using Ray head/worker nodes or PyTorch distributed training). Configuration steps are outlined in the report.

---

## Contributing

Contributions, suggestions, or improvements are welcome! If you wish to contribute:

1. **Fork** the repository.
2. Create a **feature branch** (`git checkout -b feature/your-feature`).
3. **Commit** your changes (`git commit -m 'Add your feature'`).
4. **Push** to the branch (`git push origin feature/your-feature`).
5. Open a **pull request**.

---

## License

This project is provided for educational and research purposes.
