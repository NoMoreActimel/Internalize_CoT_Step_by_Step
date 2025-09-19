import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

class GraphAnalyzer:
    """
    Comprehensive graph analysis and visualization for ProsQA datasets.
    Explains and visualizes various graph metrics and centrality measures.
    """
    
    def __init__(self, generator):
        """Initialize with a SubProsQADataset instance."""
        self.generator = generator
        self.graph = generator.graph
        self.concepts = generator.concepts
        
    def analyze_and_visualize_all(self, save_path_prefix="graph_analysis"):
        """Run all analyses and create comprehensive visualizations."""
        print("="*80)
        print("COMPREHENSIVE GRAPH ANALYSIS")
        print("="*80)
        
        # Basic statistics
        self._print_basic_stats()
        
        # Create visualizations for list/dict outputs
        self._visualize_degree_distribution(save_path_prefix)
        self._visualize_centrality_measures(save_path_prefix)
        self._visualize_clustering_coefficients(save_path_prefix)
        self._create_comprehensive_dashboard(save_path_prefix)
        
    def _print_basic_stats(self):
        """Print basic graph statistics with explanations."""
        print("\n📊 BASIC GRAPH STATISTICS")
        print("-" * 50)
        
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Graph density: {nx.density(self.graph):.4f}")
        print(f"Average clustering coefficient: {nx.average_clustering(self.graph):.4f}")
        print(f"Transitivity: {nx.transitivity(self.graph):.4f}")
        print(f"Degree assortativity: {nx.degree_assortativity_coefficient(self.graph):.4f}")
        
        # Connected components
        components = list(nx.connected_components(self.graph))
        print(f"Number of connected components: {len(components)}")
        if len(components) > 1:
            component_sizes = [len(comp) for comp in components]
            print(f"Largest component size: {max(component_sizes)}")
            print(f"Smallest component size: {min(component_sizes)}")
        
        # Path statistics
        try:
            avg_path_length = nx.average_shortest_path_length(self.graph)
            print(f"Average shortest path length: {avg_path_length:.4f}")
        except:
            print("Average shortest path length: N/A (disconnected graph)")
            
        try:
            diameter = nx.diameter(self.graph)
            print(f"Graph diameter: {diameter}")
        except:
            print("Graph diameter: N/A (disconnected graph)")
    
    def _visualize_degree_distribution(self, save_path_prefix):
        """Visualize degree distribution with explanations."""
        print("\n📈 DEGREE DISTRIBUTION ANALYSIS")
        print("-" * 50)
        print("Degree distribution shows how many nodes have each degree value.")
        print("It helps understand the graph's structure and connectivity patterns.")
        
        degrees = [d for n, d in self.graph.degree()]
        degree_hist = nx.degree_histogram(self.graph)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Degree histogram
        ax1.hist(degrees, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title('Degree Distribution Histogram')
        ax1.grid(True, alpha=0.3)
        
        # 2. Log-log plot for power law detection
        non_zero_degrees = [d for d in degrees if d > 0]
        if non_zero_degrees:
            degree_counts = {}
            for d in non_zero_degrees:
                degree_counts[d] = degree_counts.get(d, 0) + 1
            
            degrees_sorted = sorted(degree_counts.keys())
            counts_sorted = [degree_counts[d] for d in degrees_sorted]
            
            ax2.loglog(degrees_sorted, counts_sorted, 'bo-', alpha=0.7)
            ax2.set_xlabel('Degree (log scale)')
            ax2.set_ylabel('Count (log scale)')
            ax2.set_title('Degree Distribution (Log-Log Scale)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative degree distribution
        sorted_degrees = sorted(degrees, reverse=True)
        cumulative = np.cumsum(sorted_degrees) / np.sum(degrees)
        ax3.plot(range(len(cumulative)), cumulative, 'g-', linewidth=2)
        ax3.set_xlabel('Node Rank (by degree)')
        ax3.set_ylabel('Cumulative Degree Fraction')
        ax3.set_title('Cumulative Degree Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Degree statistics
        degree_stats = {
            'Mean': np.mean(degrees),
            'Median': np.median(degrees),
            'Std': np.std(degrees),
            'Min': np.min(degrees),
            'Max': np.max(degrees),
            'Skewness': self._calculate_skewness(degrees)
        }
        
        ax4.axis('off')
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in degree_stats.items()])
        ax4.text(0.1, 0.5, f'Degree Statistics:\n\n{stats_text}', 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_degree_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Mean degree: {degree_stats['Mean']:.2f}")
        print(f"Max degree: {degree_stats['Max']}")
        print(f"Degree variance: {degree_stats['Std']**2:.2f}")
    
    def _visualize_centrality_measures(self, save_path_prefix):
        """Visualize various centrality measures."""
        print("\n🎯 CENTRALITY MEASURES ANALYSIS")
        print("-" * 50)
        
        # Calculate centrality measures
        print("Calculating centrality measures...")
        degree_centrality = nx.degree_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        pagerank = nx.pagerank(self.graph)
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Degree Centrality
        self._plot_centrality_histogram(ax1, degree_centrality, 'Degree Centrality', 'blue')
        ax1.set_title('Degree Centrality\n(Proportion of nodes connected to this node)')
        
        # 2. Closeness Centrality  
        self._plot_centrality_histogram(ax2, closeness_centrality, 'Closeness Centrality', 'green')
        ax2.set_title('Closeness Centrality\n(Average distance to all other nodes)')
        
        # 3. Betweenness Centrality
        self._plot_centrality_histogram(ax3, betweenness_centrality, 'Betweenness Centrality', 'red')
        ax3.set_title('Betweenness Centrality\n(Fraction of shortest paths passing through)')
        
        # 4. PageRank
        self._plot_centrality_histogram(ax4, pagerank, 'PageRank', 'purple')
        ax4.set_title('PageRank\n(Importance based on random walk)')
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_centrality_measures.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top nodes for each centrality measure
        self._print_top_centrality_nodes(degree_centrality, closeness_centrality, 
                                       betweenness_centrality, pagerank)
        
        # Correlation analysis
        self._analyze_centrality_correlations(degree_centrality, closeness_centrality,
                                            betweenness_centrality, pagerank, save_path_prefix)
    
    def _visualize_clustering_coefficients(self, save_path_prefix):
        """Visualize clustering coefficients."""
        print("\n🔗 CLUSTERING COEFFICIENT ANALYSIS")
        print("-" * 50)
        print("Clustering coefficient measures how well connected a node's neighbors are.")
        print("High clustering indicates local community structure.")
        
        clustering_coeffs = nx.clustering(self.graph)
        clustering_values = list(clustering_coeffs.values())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Clustering coefficient histogram
        ax1.hist(clustering_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_xlabel('Clustering Coefficient')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title('Distribution of Clustering Coefficients')
        ax1.grid(True, alpha=0.3)
        
        # 2. Clustering vs Degree scatter plot
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        ax2.scatter(degrees, clustering_values, alpha=0.6, color='red')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Clustering Coefficient')
        ax2.set_title('Clustering Coefficient vs Degree')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top clustered nodes
        top_clustered = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:20]
        nodes, coeffs = zip(*top_clustered)
        ax3.bar(range(len(nodes)), coeffs, color='lightcoral')
        ax3.set_xlabel('Node Rank')
        ax3.set_ylabel('Clustering Coefficient')
        ax3.set_title('Top 20 Most Clustered Nodes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics
        stats = {
            'Mean': np.mean(clustering_values),
            'Median': np.median(clustering_values),
            'Std': np.std(clustering_values),
            'Min': np.min(clustering_values),
            'Max': np.max(clustering_values)
        }
        
        ax4.axis('off')
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in stats.items()])
        ax4.text(0.1, 0.5, f'Clustering Statistics:\n\n{stats_text}', 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_clustering_coefficients.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Average clustering coefficient: {stats['Mean']:.4f}")
        print(f"Maximum clustering coefficient: {stats['Max']:.4f}")
    
    def _create_comprehensive_dashboard(self, save_path_prefix):
        """Create a comprehensive dashboard with all measures."""
        print("\n📊 COMPREHENSIVE GRAPH DASHBOARD")
        print("-" * 50)
        
        # Calculate all measures
        degree_centrality = nx.degree_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        pagerank = nx.pagerank(self.graph)
        clustering_coeffs = nx.clustering(self.graph)
        
        # Create DataFrame for analysis
        nodes = list(self.graph.nodes())
        data = {
            'Node': nodes,
            'Degree': [self.graph.degree(n) for n in nodes],
            'Degree_Centrality': [degree_centrality[n] for n in nodes],
            'Closeness_Centrality': [closeness_centrality[n] for n in nodes],
            'Betweenness_Centrality': [betweenness_centrality[n] for n in nodes],
            'PageRank': [pagerank[n] for n in nodes],
            'Clustering_Coefficient': [clustering_coeffs[n] for n in nodes]
        }
        
        df = pd.DataFrame(data)
        
        # Create correlation heatmap
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Correlation heatmap
        corr_matrix = df[['Degree', 'Degree_Centrality', 'Closeness_Centrality', 
                         'Betweenness_Centrality', 'PageRank', 'Clustering_Coefficient']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Correlation Matrix of Graph Measures')
        
        # 2. Scatter plot: Degree vs PageRank
        ax2.scatter(df['Degree'], df['PageRank'], alpha=0.6, color='blue')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('PageRank')
        ax2.set_title('Degree vs PageRank')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot: Closeness vs Betweenness
        ax3.scatter(df['Closeness_Centrality'], df['Betweenness_Centrality'], alpha=0.6, color='red')
        ax3.set_xlabel('Closeness Centrality')
        ax3.set_ylabel('Betweenness Centrality')
        ax3.set_title('Closeness vs Betweenness Centrality')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top nodes summary
        top_nodes = df.nlargest(10, 'PageRank')[['Node', 'Degree', 'PageRank', 'Clustering_Coefficient']]
        ax4.axis('off')
        table_text = top_nodes.to_string(index=False, float_format='%.4f')
        ax4.text(0.05, 0.95, f'Top 10 Nodes by PageRank:\n\n{table_text}', 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_centrality_histogram(self, ax, centrality_dict, title, color):
        """Helper function to plot centrality histogram."""
        values = list(centrality_dict.values())
        ax.hist(values, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel(title)
        ax.set_ylabel('Number of Nodes')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    def _print_top_centrality_nodes(self, degree_cent, closeness_cent, betweenness_cent, pagerank):
        """Print top nodes for each centrality measure."""
        print("\nTop 5 nodes by each centrality measure:")
        print("-" * 40)
        
        measures = [
            ('Degree Centrality', degree_cent),
            ('Closeness Centrality', closeness_cent),
            ('Betweenness Centrality', betweenness_cent),
            ('PageRank', pagerank)
        ]
        
        for measure_name, measure_dict in measures:
            top_nodes = sorted(measure_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n{measure_name}:")
            for node, value in top_nodes:
                concept_name = self.concepts[node] if node < len(self.concepts) else f"Node_{node}"
                print(f"  {concept_name}: {value:.4f}")
    
    def _analyze_centrality_correlations(self, degree_cent, closeness_cent, betweenness_cent, pagerank, save_path_prefix):
        """Analyze correlations between centrality measures."""
        print("\n🔗 CENTRALITY CORRELATIONS")
        print("-" * 30)
        
        # Create correlation plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        measures = {
            'Degree': degree_cent,
            'Closeness': closeness_cent,
            'Betweenness': betweenness_cent,
            'PageRank': pagerank
        }
        
        # Calculate correlations
        nodes = list(self.graph.nodes())
        corr_data = {}
        for name, measure in measures.items():
            corr_data[name] = [measure[n] for n in nodes]
        
        corr_df = pd.DataFrame(corr_data)
        corr_matrix = corr_df.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Centrality Measures Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_centrality_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print correlation insights
        print("Key correlations:")
        print(f"  Degree ↔ PageRank: {corr_matrix.loc['Degree', 'PageRank']:.3f}")
        print(f"  Closeness ↔ Betweenness: {corr_matrix.loc['Closeness', 'Betweenness']:.3f}")
        print(f"  Degree ↔ Closeness: {corr_matrix.loc['Degree', 'Closeness']:.3f}")
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
    
    def explain_measures(self):
        """Print detailed explanations of all graph measures."""
        print("\n" + "="*80)
        print("DETAILED EXPLANATION OF GRAPH MEASURES")
        print("="*80)
        
        explanations = {
            "Degree Distribution": """
            The degree distribution shows how many nodes have each degree value.
            - Degree = number of connections a node has
            - Helps identify if the graph follows a power law (scale-free)
            - Power law graphs have few highly connected "hub" nodes
            - Random graphs have Poisson-like degree distributions
            """,
            
            "Degree Centrality": """
            Degree centrality measures the number of direct connections a node has.
            - Formula: degree(node) / (total_nodes - 1)
            - Range: [0, 1]
            - High value = well-connected node
            - Simple but effective measure of local importance
            """,
            
            "Closeness Centrality": """
            Closeness centrality measures how close a node is to all other nodes.
            - Formula: (n-1) / sum(shortest_path_lengths_to_all_other_nodes)
            - Range: [0, 1]
            - High value = node can reach others quickly
            - Good for identifying "central" nodes in communication networks
            """,
            
            "Betweenness Centrality": """
            Betweenness centrality measures how often a node lies on shortest paths.
            - Formula: sum(shortest_paths_through_node / total_shortest_paths)
            - Range: [0, 1]
            - High value = node is a "bridge" or "bottleneck"
            - Important for identifying critical nodes in network flow
            """,
            
            "PageRank": """
            PageRank measures node importance based on random walk behavior.
            - Considers both quantity and quality of connections
            - Damping factor prevents random walk from getting stuck
            - Range: [0, 1] (sums to 1 across all nodes)
            - High value = node is "important" in the network
            - Used by Google for web page ranking
            """,
            
            "Clustering Coefficient": """
            Clustering coefficient measures local community structure.
            - Formula: actual_edges_between_neighbors / possible_edges_between_neighbors
            - Range: [0, 1]
            - High value = node's neighbors are well-connected to each other
            - Indicates presence of local communities or cliques
            - Global average = transitivity of the entire graph
            """,
            
            "Transitivity": """
            Transitivity measures the overall clustering tendency of the graph.
            - Formula: 3 * number_of_triangles / number_of_connected_triples
            - Range: [0, 1]
            - High value = graph has many triangles (A-B-C-A patterns)
            - Indicates strong community structure
            """,
            
            "Degree Assortativity": """
            Degree assortativity measures whether nodes connect to similar-degree nodes.
            - Range: [-1, 1]
            - Positive = high-degree nodes connect to other high-degree nodes
            - Negative = high-degree nodes connect to low-degree nodes
            - Zero = random mixing
            - Affects network resilience and information flow
            """
        }
        
        for measure, explanation in explanations.items():
            print(f"\n📊 {measure}")
            print("-" * len(measure) - 2)
            print(explanation.strip())


# Example usage
if __name__ == "__main__":
    from generate_shared_graph_cot_2 import SubProsQADataset
    
    # Create a sample dataset
    dataset = SubProsQADataset(
        num_nodes=500,
        num_edges=1000,
        num_samples=100,
        num_context_edges=20,
        representation='structured',
        candidate_samples_path='candidate_samples.json',
        load_candidate_samples=False,
        dataset_path='prosqa_dataset.json',
        load_dataset=False,
        depth_range=(3, 6),
        seed=42
    )
    
    # Create analyzer and run comprehensive analysis
    analyzer = GraphAnalyzer(dataset)
    
    # Print explanations
    analyzer.explain_measures()
    
    # Run all analyses and visualizations
    analyzer.analyze_and_visualize_all("prosqa_graph_analysis")
