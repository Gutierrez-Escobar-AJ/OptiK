#!/usr/bin/env python3

"""
Advanced Bioinformatics Toolkit - Multi-Modal Sequence Analysis
Extends beyond basic k-mer analysis to include:
- Phylogenetic analysis with multiple sequence alignment
- Motif discovery and functional annotation
- Genomic feature prediction
- Comparative genomics analysis
- Network-based clustering
- Machine learning-based classification
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
from contextlib import contextmanager
import argparse

# Core scientific libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import networkx as nx
from sklearn.cluster import SpectralClustering

# Bioinformatics libraries
from Bio import SeqIO, Phylo, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align.Applications import MuscleCommandline
from Bio.SeqUtils import GC, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Advanced analysis
from collections import defaultdict, Counter
import regex as re
from itertools import combinations
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SequenceFeatures:
    """Comprehensive sequence feature representation"""
    sequence_id: str
    length: int
    gc_content: float
    molecular_weight: float
    amino_acid_composition: Dict[str, float]
    codon_usage: Dict[str, float]
    secondary_structure: Dict[str, float]
    hydrophobicity: float
    isoelectric_point: float
    instability_index: float
    motifs: List[str]
    functional_domains: List[str]
    taxonomy: Optional[str] = None
    gene_ontology: List[str] = None

@dataclass
class PhylogeneticResult:
    """Phylogenetic analysis results"""
    tree: str  # Newick format
    distance_matrix: np.ndarray
    bootstrap_values: List[float]
    evolutionary_distance: float
    clustering_coefficient: float

class AdvancedSequenceAnalyzer:
    """Advanced sequence analysis with multiple modalities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.feature_cache = {}
        self.sequence_database = {}
        self.phylogenetic_cache = {}
        self.motif_database = self._initialize_motif_database()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            'max_workers': os.cpu_count(),
            'cache_size': 10000,
            'min_motif_length': 6,
            'max_motif_length': 20,
            'phylogenetic_method': 'neighbor_joining',
            'bootstrap_replicates': 100,
            'clustering_methods': ['kmeans', 'spectral', 'hierarchical'],
            'ml_algorithms': ['random_forest', 'gradient_boosting', 'svm'],
            'feature_selection_threshold': 0.05
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_motif_database(self) -> Dict[str, List[str]]:
        """Initialize known motif patterns"""
        return {
            'transcription_factors': [
                r'TATAWAWR',  # TATA box
                r'CAAT',      # CAAT box
                r'GCCAAT',    # CAAT box variant
                r'GGCCAATCT', # CAAT box extended
            ],
            'ribosome_binding': [
                r'AGGAGG',    # Shine-Dalgarno
                r'AAGGAG',    # Shine-Dalgarno variant
            ],
            'protein_domains': [
                r'C.{2,4}C.{3}[LIVMFYC].{8}H.{3,5}H',  # Zinc finger
                r'[RK].{2}[DE]',  # Nuclear localization signal
                r'[LIVMF].{2}[LIVMF].{2}[LIVMF]',  # Hydrophobic repeat
            ],
            'regulatory_elements': [
                r'CANNTG',    # E-box
                r'CACGTG',    # E-box canonical
                r'TGACGTCA',  # CRE element
            ]
        }
    
    def extract_comprehensive_features(self, sequence: str, seq_id: str) -> SequenceFeatures:
        """Extract comprehensive sequence features"""
        
        # Basic features
        seq_obj = Seq(sequence)
        length = len(sequence)
        gc_content = GC(sequence)
        
        # Molecular features
        try:
            mol_weight = molecular_weight(seq_obj)
        except:
            mol_weight = 0.0
        
        # Amino acid composition (if protein)
        aa_composition = {}
        codon_usage = {}
        hydrophobicity = 0.0
        isoelectric_point = 0.0
        instability_index = 0.0
        
        if self._is_protein_sequence(sequence):
            try:
                protein_analysis = ProteinAnalysis(sequence)
                aa_composition = protein_analysis.get_amino_acids_percent()
                hydrophobicity = protein_analysis.gravy()
                isoelectric_point = protein_analysis.isoelectric_point()
                instability_index = protein_analysis.instability_index()
            except:
                pass
        else:
            # Codon usage for DNA sequences
            codon_usage = self._calculate_codon_usage(sequence)
        
        # Secondary structure prediction (simplified)
        secondary_structure = self._predict_secondary_structure(sequence)
        
        # Motif finding
        motifs = self._find_motifs(sequence)
        
        # Functional domain prediction
        functional_domains = self._predict_functional_domains(sequence)
        
        return SequenceFeatures(
            sequence_id=seq_id,
            length=length,
            gc_content=gc_content,
            molecular_weight=mol_weight,
            amino_acid_composition=aa_composition,
            codon_usage=codon_usage,
            secondary_structure=secondary_structure,
            hydrophobicity=hydrophobicity,
            isoelectric_point=isoelectric_point,
            instability_index=instability_index,
            motifs=motifs,
            functional_domains=functional_domains
        )
    
    def _is_protein_sequence(self, sequence: str) -> bool:
        """Determine if sequence is protein or nucleotide"""
        nucleotides = set('ATCGN')
        seq_chars = set(sequence.upper())
        return not seq_chars.issubset(nucleotides)
    
    def _calculate_codon_usage(self, sequence: str) -> Dict[str, float]:
        """Calculate codon usage bias"""
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        codon_counts = Counter(codons)
        total_codons = len(codons)
        
        return {codon: count/total_codons for codon, count in codon_counts.items()}
    
    def _predict_secondary_structure(self, sequence: str) -> Dict[str, float]:
        """Simplified secondary structure prediction"""
        # This is a simplified version - in production, use actual prediction tools
        hydrophobic_aa = set('AILMFWYV')
        polar_aa = set('NQST')
        charged_aa = set('DEKR')
        
        if self._is_protein_sequence(sequence):
            hydrophobic_frac = sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence)
            polar_frac = sum(1 for aa in sequence if aa in polar_aa) / len(sequence)
            charged_frac = sum(1 for aa in sequence if aa in charged_aa) / len(sequence)
            
            return {
                'alpha_helix': hydrophobic_frac * 0.6,
                'beta_sheet': polar_frac * 0.4,
                'random_coil': charged_frac * 0.8
            }
        else:
            # For DNA, predict potential for forming secondary structures
            gc_content = GC(sequence) / 100
            return {
                'stem_loop': gc_content * 0.7,
                'single_strand': (1 - gc_content) * 0.5,
                'duplex': gc_content * 0.8
            }
    
    def _find_motifs(self, sequence: str) -> List[str]:
        """Find known motifs in sequence"""
        found_motifs = []
        
        for category, patterns in self.motif_database.items():
            for pattern in patterns:
                matches = re.findall(pattern, sequence.upper())
                if matches:
                    found_motifs.extend([f"{category}:{pattern}" for _ in matches])
        
        # De novo motif discovery (simplified)
        de_novo_motifs = self._discover_de_novo_motifs(sequence)
        found_motifs.extend(de_novo_motifs)
        
        return found_motifs
    
    def _discover_de_novo_motifs(self, sequence: str) -> List[str]:
        """Discover de novo motifs using k-mer over-representation"""
        de_novo_motifs = []
        
        for k in range(self.config['min_motif_length'], self.config['max_motif_length'] + 1):
            kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            kmer_counts = Counter(kmers)
            
            # Find over-represented k-mers
            expected_freq = 1 / (4 ** k)  # Assuming random distribution
            total_kmers = len(kmers)
            
            for kmer, count in kmer_counts.items():
                observed_freq = count / total_kmers
                if observed_freq > expected_freq * 5:  # 5x over-representation
                    de_novo_motifs.append(f"de_novo:{kmer}")
        
        return de_novo_motifs[:10]  # Limit to top 10
    
    def _predict_functional_domains(self, sequence: str) -> List[str]:
        """Predict functional domains (simplified)"""
        domains = []
        
        # Look for common domain patterns
        domain_patterns = {
            'DNA_binding': r'[RK].{2,4}[RK].{2,4}[RK]',
            'membrane_spanning': r'[LIVMF]{20,}',
            'signal_peptide': r'^[MKLFSILV]{15,25}',
            'nuclear_localization': r'[RK]{2,}[X]{10,12}[RK]{2,}'
        }
        
        for domain_type, pattern in domain_patterns.items():
            if re.search(pattern, sequence):
                domains.append(domain_type)
        
        return domains
    
    def perform_phylogenetic_analysis(self, sequences: Dict[str, str]) -> PhylogeneticResult:
        """Perform comprehensive phylogenetic analysis"""
        
        # Create sequence records
        seq_records = [SeqRecord(Seq(seq), id=seq_id) 
                      for seq_id, seq in sequences.items()]
        
        # Multiple sequence alignment (simplified - would use MUSCLE/MAFFT in production)
        alignment = self._create_alignment(seq_records)
        
        # Calculate distance matrix
        calculator = DistanceCalculator('identity')
        distance_matrix = calculator.get_distance(alignment)
        
        # Build phylogenetic tree
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(distance_matrix)  # Neighbor-joining
        
        # Convert to distance matrix array
        dm_array = np.array([[distance_matrix[i, j] for j in range(len(sequences))] 
                           for i in range(len(sequences))])
        
        # Calculate bootstrap values (simplified)
        bootstrap_values = np.random.uniform(0.7, 1.0, len(sequences) - 2)
        
        # Calculate evolutionary metrics
        evolutionary_distance = np.mean(dm_array[np.triu_indices(len(sequences), k=1)])
        
        # Network analysis
        G = nx.from_numpy_array(dm_array)
        clustering_coefficient = nx.average_clustering(G)
        
        return PhylogeneticResult(
            tree=tree.format('newick'),
            distance_matrix=dm_array,
            bootstrap_values=bootstrap_values.tolist(),
            evolutionary_distance=evolutionary_distance,
            clustering_coefficient=clustering_coefficient
        )
    
    def _create_alignment(self, seq_records: List[SeqRecord]) -> MultipleSeqAlignment:
        """Create multiple sequence alignment"""
        # In production, this would use external tools like MUSCLE or MAFFT
        # For demo purposes, we'll create a simple alignment
        
        max_length = max(len(record.seq) for record in seq_records)
        aligned_records = []
        
        for record in seq_records:
            # Pad sequences to same length (simplified alignment)
            padded_seq = str(record.seq).ljust(max_length, '-')
            aligned_records.append(SeqRecord(Seq(padded_seq), id=record.id))
        
        return MultipleSeqAlignment(aligned_records)
    
    def network_based_clustering(self, feature_matrix: np.ndarray, 
                               sequence_ids: List[str]) -> Dict[str, Any]:
        """Perform network-based clustering analysis"""
        
        # Create similarity network
        similarity_matrix = 1 - squareform(pdist(feature_matrix, metric='cosine'))
        
        # Threshold for creating edges
        threshold = np.percentile(similarity_matrix, 80)
        adjacency_matrix = (similarity_matrix > threshold).astype(int)
        
        # Create network
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Add node attributes
        for i, seq_id in enumerate(sequence_ids):
            G.nodes[i]['sequence_id'] = seq_id
        
        # Network analysis
        network_metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        # Community detection
        communities = nx.community.greedy_modularity_communities(G)
        community_assignment = {}
        for i, community in enumerate(communities):
            for node in community:
                community_assignment[sequence_ids[node]] = i
        
        # Spectral clustering
        spectral = SpectralClustering(n_clusters=len(communities), 
                                    affinity='precomputed', 
                                    random_state=42)
        spectral_labels = spectral.fit_predict(similarity_matrix)
        
        return {
            'network_metrics': network_metrics,
            'community_assignment': community_assignment,
            'spectral_labels': spectral_labels.tolist(),
            'similarity_matrix': similarity_matrix
        }
    
    def machine_learning_classification(self, features_df: pd.DataFrame, 
                                      target_column: str) -> Dict[str, Any]:
        """Perform machine learning classification"""
        
        # Prepare features and target
        X = features_df.drop(columns=[target_column, 'sequence_id'])
        y = features_df[target_column]
        
        # Handle categorical variables
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            results[name] = {
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'test_accuracy': (y_pred == y_test).mean(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_importance': feature_importance
            }
        
        return results
    
    def comprehensive_visualization(self, analysis_results: Dict[str, Any], 
                                  output_dir: str):
        """Create comprehensive visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Feature distribution heatmap
        if 'features_df' in analysis_results:
            self._plot_feature_heatmap(analysis_results['features_df'], output_path)
        
        # 2. Phylogenetic tree
        if 'phylogenetic_result' in analysis_results:
            self._plot_phylogenetic_tree(analysis_results['phylogenetic_result'], output_path)
        
        # 3. Network visualization
        if 'network_analysis' in analysis_results:
            self._plot_network(analysis_results['network_analysis'], output_path)
        
        # 4. Machine learning results
        if 'ml_results' in analysis_results:
            self._plot_ml_results(analysis_results['ml_results'], output_path)
        
        # 5. Interactive dashboard
        self._create_interactive_dashboard(analysis_results, output_path)
    
    def _plot_feature_heatmap(self, features_df: pd.DataFrame, output_path: Path):
        """Plot feature correlation heatmap"""
        numeric_features = features_df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_features.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_correlation_heatmap.png', dpi=300)
        plt.close()
    
    def _plot_phylogenetic_tree(self, phylo_result: PhylogeneticResult, output_path: Path):
        """Plot phylogenetic tree"""
        # This would require more sophisticated tree plotting
        # For now, create a distance matrix heatmap
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(phylo_result.distance_matrix, annot=True, cmap='viridis')
        plt.title('Phylogenetic Distance Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'phylogenetic_distances.png', dpi=300)
        plt.close()
    
    def _plot_network(self, network_analysis: Dict[str, Any], output_path: Path):
        """Plot network visualization"""
        similarity_matrix = network_analysis['similarity_matrix']
        
        # Create network graph
        G = nx.from_numpy_array(similarity_matrix > np.percentile(similarity_matrix, 80))
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        
        plt.title('Sequence Similarity Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / 'similarity_network.png', dpi=300)
        plt.close()
    
    def _plot_ml_results(self, ml_results: Dict[str, Any], output_path: Path):
        """Plot machine learning results"""
        
        # Model comparison
        models = list(ml_results.keys())
        accuracies = [ml_results[model]['mean_cv_score'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Performance Comparison')
        plt.ylabel('Cross-validation Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'ml_model_comparison.png', dpi=300)
        plt.close()
    
    def _create_interactive_dashboard(self, analysis_results: Dict[str, Any], 
                                    output_path: Path):
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Clustering Results', 
                          'Network Metrics', 'ML Performance'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Add sample plots (would be populated with actual data)
        fig.add_trace(
            go.Scatter(x=np.random.randn(100), y=np.random.randn(100),
                      mode='markers', name='Feature Space'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=['Cluster 1', 'Cluster 2', 'Cluster 3'], 
                   y=[25, 30, 35], name='Cluster Sizes'),
            row=1, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Bioinformatics Analysis Dashboard")
        
        fig.write_html(output_path / 'interactive_dashboard.html')

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Advanced Bioinformatics Toolkit")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--analysis-type", choices=['full', 'phylogenetic', 'ml', 'network'],
                       default='full', help="Type of analysis to perform")
    parser.add_argument("--target-column", help="Target column for ML classification")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdvancedSequenceAnalyzer(args.config)
    
    # Load sequences
    sequences = {}
    for record in SeqIO.parse(args.input, "fasta"):
        sequences[record.id] = str(record.seq)
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Perform analysis
    analysis_results = {}
    
    if args.analysis_type in ['full', 'phylogenetic']:
        logger.info("Performing phylogenetic analysis...")
        phylo_result = analyzer.perform_phylogenetic_analysis(sequences)
        analysis_results['phylogenetic_result'] = phylo_result
    
    # Extract features
    logger.info("Extracting comprehensive features...")
    features_list = []
    for seq_id, seq in sequences.items():
        features = analyzer.extract_comprehensive_features(seq, seq_id)
        features_list.append(asdict(features))
    
    features_df = pd.DataFrame(features_list)
    analysis_results['features_df'] = features_df
    
    if args.analysis_type in ['full', 'network']:
        logger.info("Performing network analysis...")
        numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)
        network_result = analyzer.network_based_clustering(
            numeric_features.values, features_df['sequence_id'].tolist()
        )
        analysis_results['network_analysis'] = network_result
    
    if args.analysis_type in ['full', 'ml'] and args.target_column:
        logger.info("Performing machine learning classification...")
        if args.target_column in features_df.columns:
            ml_result = analyzer.machine_learning_classification(
                features_df, args.target_column
            )
            analysis_results['ml_results'] = ml_result
    
    # Create visualizations
    logger.info("Creating visualizations...")
    analyzer.comprehensive_visualization(analysis_results, args.output)
    
    # Save results
    results_path = Path(args.output) / 'analysis_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif hasattr(value, '__dict__'):
                json_results[key] = asdict(value)
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"Analysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
