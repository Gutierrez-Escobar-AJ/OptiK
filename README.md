# OptiK: An Entropy-Driven Framework for k-mer Size Optimization in Comparative Genomics

<p align="center">
  <img src="Logo.png" alt="ERICLoD Logo" width="350"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OptiK** is a reproducible, unsupervised, and alignment-free tool for selecting the optimal k-mer size for comparative genomic analyses.  
It leverages entropy-aware representations, dimensionality reduction, and clustering validation metrics to identify the k-mer length that captures the most informative structure in a given genome dataset.

---

## ✨ Features

- Canonicalized k-mer frequency matrix construction
- Truncated SVD for dimensionality reduction
- Unsupervised clustering (KMeans or Agglomerative)
- Evaluation with:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
- Rank-based consensus selection of optimal `k`
- Optional UMAP and dendrogram visualizations
- Reproducible and configurable command-line interface

---

## 🧬 Use Cases

- Preprocessing for genome clustering, diversity estimation, or taxonomic resolution
- Selecting optimal `k` for Kraken, Mash, sourmash, or other k-mer-based tools
- Visual inspection of signal structure in comparative genomics or metagenomics

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Gutierrez-Escobar-AJ/OptiK.git
cd OptiK

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, use the included `environment.yml` to create a Conda environment.

---

## 🚀 Usage

```bash
python3 optik_v2.py \
  -i ./data/genomes/ \
  -o ./results/ \
  --clusterer kmeans \
  --k-range 3 8 \
  --plot-umap
```

---

### ✅ Example Output

- `metrics.csv` — validation scores (Silhouette, CH, DB) per k and cluster count
- `best_k.txt` — the selected optimal k-mer size
- `cluster_assignments_k8.csv` — genome-to-cluster mapping
- `umap_k8.png` — 2D UMAP of final clustering structure

---

## 📄 Input Requirements

- Input: Directory of genome FASTA files
- Optional: Metadata CSV with known subpopulation labels (for visual validation)

---

## 📊 Metric-Based Optimization

For each `k`, OptiK:
- Computes clustering metrics across a user-defined cluster range (default: 3–8)
- Ranks each `k` independently within each metric
- Selects the optimal `k` based on lowest total rank
- Resolves ties by favoring higher Silhouette and lower Davies-Bouldin scores

---

## 🔁 Reproducibility

- Set `--random-seed` for deterministic runs
- All inputs, outputs, and configuration are stored for full reproducibility

---

## 📚 Citation

If you use OptiK in your research, please cite:

```
Gutierrez Escobar AJ. OptiK: An Entropy-Driven Framework for Optimal k-mer Size Selection in Comparative Genomics. bioRxiv (2025).
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Acknowledgments

OptiK builds upon excellent libraries including:
- NumPy
- scikit-learn
- Biopython
- UMAP-learn

---

For test datasets, example notebooks, and a full walkthrough of the HpGP use case, see the `notebooks/` folder or visit the [Wiki](https://github.com/yourusername/OptiK/wiki).
