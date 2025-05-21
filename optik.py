#!/usr/bin/env python3

"""
OptiK - Final Corrected Version (All bugs fixed, warnings suppressed, numpy-safe)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import gc
import argparse
import logging
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Dict, Any

import psutil
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["NUMBA_DISABLE_WARNINGS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("optik.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('OptiK')

def count_kmers(sequence: str, k: int, min_freq: int = 1) -> Dict[str, float]:
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    counts = Counter(kmers)
    if min_freq > 1:
        counts = {mer: cnt for mer, cnt in counts.items() if cnt >= min_freq}
    total = sum(counts.values())
    return {mer: cnt/total for mer, cnt in counts.items()} if total else {}

def process_file(filepath: str, k: int, min_freq: int) -> Tuple[List[Dict[str, float]], List[str]]:
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        vectors, names = [], []
        for rec in records:
            vec = count_kmers(str(rec.seq).upper(), k, min_freq)
            if vec:
                vectors.append(vec)
                names.append(rec.id)
        return vectors, names
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return [], []

def plot_umap(reduced: np.ndarray, labels: List[int], k: int, out_dir: str):
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(np.array(reduced))
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=np.array(labels), cmap='tab10', s=10)
    plt.title(f"UMAP Projection (k={k})")
    plt.colorbar(scatter, label="Cluster ID")
    plt.savefig(os.path.join(out_dir, f"umap_k{k}.png"))
    plt.close()

def plot_dendrogram(reduced: np.ndarray, k: int, out_dir: str):
    dist = pdist(np.array(reduced))
    Z = linkage(dist, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, no_labels=True)
    plt.title(f"Dendrogram (k={k})")
    plt.savefig(os.path.join(out_dir, f"dendrogram_k{k}.png"))
    plt.close()

class OptiK:
    def __init__(self, args):
        self.args = args
        self.results: List[Dict[str, Any]] = []
        self._validate_args()
        self._check_memory()
        os.makedirs(self.args.output, exist_ok=True)

    def _validate_args(self):
        if not os.path.isdir(self.args.input):
            raise FileNotFoundError(f"Input directory not found: {self.args.input}")
        if any(k < 1 for k in self.args.k_values):
            raise ValueError("All k-values must be ≥ 1")
        if self.args.clusterer not in ['kmeans', 'agg']:
            raise ValueError("Clusterer must be one of: kmeans, agg")
        if any(k > 8 for k in self.args.k_values):
            logger.warning("⚠️ High k-mer sizes (>8) may cause memory exhaustion!")

    def _check_memory(self):
        avail = psutil.virtual_memory().available / (1024**3)
        if self.args.max_memory > avail:
            logger.warning(f"Requested memory ({self.args.max_memory}GB) exceeds available ({avail:.2f}GB)")

    def _process_batch(self, files: List[str], k: int) -> Tuple[np.ndarray, List[str]]:
        with Pool(min(cpu_count(), len(files))) as pool:
            batches = list(tqdm(pool.imap(partial(process_file, k=k, min_freq=self.args.min_kmer_freq), files),
                                total=len(files), desc=f"Processing k={k}"))
        vects, names = [], []
        for vec_list, nm_list in batches:
            vects.extend(vec_list)
            names.extend(nm_list)
        if not vects:
            return np.array([]), []
        all_kmers = sorted({mer for vec in vects for mer in vec})
        idx = {mer:i for i, mer in enumerate(all_kmers)}
        mat = np.zeros((len(vects), len(all_kmers)), dtype=np.float32)
        for i, vec in enumerate(vects):
            for mer, val in vec.items():
                mat[i, idx[mer]] = val
        return mat, names

    def _cluster_and_metrics(self, data: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        metrics: Dict[str, float] = {}
        labels = None
        if self.args.clusterer == 'kmeans':
            for n in range(3, 9):
                try:
                    km = KMeans(n_clusters=n, n_init=10).fit(data)
                    lbl = np.array(km.labels_)
                    metrics[f"silhouette_{n}"] = silhouette_score(data, lbl)
                    metrics[f"calinski_{n}"] = calinski_harabasz_score(data, lbl)
                    metrics[f"davies_{n}"] = davies_bouldin_score(data, lbl)
                    if n == 8:
                        labels = lbl
                except Exception as e:
                    logger.warning(f"KMeans failed for {n} clusters: {e}")
        else:
            try:
                model = AgglomerativeClustering()
                lbl = np.array(model.fit_predict(data))
                metrics["silhouette"] = silhouette_score(data, lbl)
                metrics["calinski"] = calinski_harabasz_score(data, lbl)
                metrics["davies"] = davies_bouldin_score(data, lbl)
                labels = lbl
            except Exception as e:
                logger.error(f"AgglomerativeClustering failed: {e}")
        return metrics, labels

    def _save_progress(self):
        pd.DataFrame(self.results).to_csv(os.path.join(self.args.output, "progress.csv"), index=False)

    def _plot_metrics(self, df: pd.DataFrame):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        x = df['k'].to_numpy()
        if self.args.clusterer == 'kmeans':
            for n in range(3, 9):
                for metric, ax in zip(['silhouette', 'calinski', 'davies'], axes):
                    col = f"{metric}_{n}"
                    if col in df:
                        y = df[col].to_numpy()
                        ax.plot(x, y, label=f"{n} clusters")
        else:
            for i, metric in enumerate(['silhouette', 'calinski', 'davies']):
                if metric in df:
                    y = df[metric].to_numpy()
                    axes[i].plot(x, y, marker='o')
        for i, (ax, title) in enumerate(zip(axes, ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'])):
            ax.set_title(title)
            if self.args.clusterer == 'kmeans':
                ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output, "metrics.png"))
        plt.close()

    def _generate_output(self):
        if not self.results:
            logger.warning("No results to save")
            return
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.args.output, "final_results.csv"), index=False)
        if not self.args.no_plots:
            self._plot_metrics(df)

    def _analyze_k(self, k: int) -> Dict[str, Any]:
        logger.info(f"Analyzing k={k}")
        files = [os.path.join(self.args.input, f)
                 for f in os.listdir(self.args.input)
                 if f.lower().endswith(('.fasta', '.fa', '.fna'))]
        mat, names = self._process_batch(files, k)
        if mat.size == 0:
            logger.warning(f"No valid sequences for k={k}")
            return {}
        svd = TruncatedSVD(n_components=min(50, mat.shape[1] - 1))
        reduced = np.array(svd.fit_transform(mat))
        metrics, labels = self._cluster_and_metrics(reduced)
        np.save(os.path.join(self.args.output, f"k{k}_matrix.npy"), mat)
        if self.args.plot_umap and labels is not None:
            plot_umap(reduced, labels, k, self.args.output)
        if self.args.plot_dendro and self.args.clusterer == 'agg':
            plot_dendrogram(reduced, k, self.args.output)
        return {'k': k, **metrics}

    def run(self):
        try:
            for k in self.args.k_values:
                out = self._analyze_k(k)
                if out:
                    self.results.append(out)
                    self._save_progress()
            self._generate_output()
            logger.info("Analysis completed successfully")
        except Exception as e:
            logger.critical(f"Pipeline failed: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="OptiK: Optimized K-mer Analysis Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Directory with FASTA files")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-k", "--k-values", type=int, nargs="+", default=list(range(3, 9)),
                        help="k-mer sizes to evaluate (default: 3 to 8)")
    parser.add_argument("--clusterer", choices=['kmeans', 'agg'], default='kmeans',
                        help="Clustering method: kmeans or agg")
    parser.add_argument("--min-kmer-freq", type=int, default=1, help="Min frequency to retain k-mer")
    parser.add_argument("--max-memory", type=float, default=32.0, help="Max memory (GB)")
    parser.add_argument("--no-plots", action="store_true", help="Disable metrics plots")
    parser.add_argument("--plot-umap", action="store_true", help="Enable UMAP plots")
    parser.add_argument("--plot-dendro", action="store_true", help="Enable dendrogram plots")
    args = parser.parse_args()
    OptiK(args).run()

if __name__ == "__main__":
    main()
    
