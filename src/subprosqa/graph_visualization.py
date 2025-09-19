import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import random
from typing import List, Tuple, Dict, Set

class ProsQAGraphVisualizer:
    """
    Visualize ProsQA graphs with highlighted context edges, reasoning paths,
    and question nodes.
    """

    def __init__(self, generator):
        """Initialize with a ProsQAGenerator instance."""
        self.generator = generator
        self.graph = generator.graph
        self.concepts = generator.concepts

        # Color scheme
        self.colors = {
            'context_edge': '#FF6B6B',      # Red for context edges
            'reasoning_path': '#4ECDC4',     # Teal for reasoning path
            'path_and_context': '#8A2BE2',  # Purple for edges both on path and in context
            'other_edge': '#E8E8E8',        # Light gray for other edges
            'source_node': '#FFD93D',       # Yellow for source
            'reachable_target': '#6BCF7F',   # Green for reachable target
            'unreachable_target': '#FF6B6B', # Red for unreachable target
            'path_node': '#A8E6CF',         # Light green for path nodes
            'other_node': '#F0F0F0'         # Light gray for other nodes
        }

    def visualize_sample(self, sample_data=None, figsize=(16, 12),
                        layout='spring', node_size_scale=1.0,
                        show_labels=True, label_font_size=8,
                        edge_width_scale=1.0, save_path=None):
        """
        Visualize a single sample showing context edges, reasoning path, and key nodes.

        Args:
            sample_data: Dict containing sample info, or None to generate new sample
            figsize: Figure size tuple
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell')
            node_size_scale: Scale factor for node sizes
            show_labels: Whether to show node labels
            label_font_size: Font size for labels
            edge_width_scale: Scale factor for edge widths
            save_path: Path to save figure (optional)
        """

        # Extract information from sample
        source_concept = sample_data['source']
        reachable_concept = sample_data['reachable_target']
        unreachable_concept = sample_data['unreachable_target']
        reasoning_path = sample_data['path']

        # Get node indices
        source_idx = self.concepts.index(source_concept)
        reachable_idx = self.concepts.index(reachable_concept)
        unreachable_idx = self.concepts.index(unreachable_concept)
        path_indices = [self.concepts.index(concept) for concept in reasoning_path]

        # Get context edges from the structured context
        context_edges = self._extract_context_edges(sample_data['context'])

        # Create subgraph containing relevant nodes
        relevant_nodes = set()
        relevant_nodes.update(path_indices)
        relevant_nodes.add(unreachable_idx)

        # Add nodes connected by context edges
        for u, v in context_edges:
            relevant_nodes.add(u)
            relevant_nodes.add(v)

        # Add some random neighboring nodes for context
        for node in list(relevant_nodes)[:]:
            neighbors = list(self.graph.predecessors(node)) + list(self.graph.successors(node))
            relevant_nodes.update(random.sample(neighbors, min(3, len(neighbors))))

        # Limit graph size for visualization
        if len(relevant_nodes) > 100:
            # Keep most important nodes
            important_nodes = set(path_indices + [unreachable_idx])
            context_nodes = set()
            for u, v in context_edges:
                context_nodes.update([u, v])

            other_nodes = relevant_nodes - important_nodes - context_nodes
            other_nodes = set(random.sample(list(other_nodes),
                                          min(50, len(other_nodes))))

            relevant_nodes = important_nodes | context_nodes | other_nodes

        # Create subgraph
        subgraph = self.graph.subgraph(relevant_nodes)

        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Full context visualization
        self._plot_graph(subgraph, ax1, source_idx, reachable_idx, unreachable_idx,
                        path_indices, context_edges, layout, node_size_scale,
                        show_labels, label_font_size, edge_width_scale,
                        title="Graph Visualization with Context")

        # Right plot: Legend and sample information
        self._plot_sample_info(ax2, sample_data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

    def _extract_context_edges(self, context_str):
        """Extract edge tuples from structured context string."""
        edges = []
        if context_str.startswith('[EDGES]'):
            edge_part = context_str[7:].strip()
            if '[FACTS]' in edge_part:
                edge_part = edge_part.split('[FACTS]')[0].strip()

            edge_strings = [e.strip() for e in edge_part.split(',') if '->' in e]

            for edge_str in edge_strings:
                if '->' in edge_str:
                    u_name, v_name = edge_str.split('->')
                    u_name, v_name = u_name.strip(), v_name.strip()
                    try:
                        u_idx = self.concepts.index(u_name)
                        v_idx = self.concepts.index(v_name)
                        edges.append((u_idx, v_idx))
                    except ValueError:
                        continue

        return edges

    def _plot_graph(self, graph, ax, source_idx, reachable_idx, unreachable_idx,
                   path_indices, context_edges, layout, node_size_scale,
                   show_labels, label_font_size, edge_width_scale, title):
        """Plot the graph with highlighting."""

        # Generate layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=3, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)

        # Categorize edges
        context_edge_set = set(context_edges)
        reasoning_edges = set(zip(path_indices[:-1], path_indices[1:]))

        # Find edges that are both on the path and in context
        path_and_context_edges = reasoning_edges.intersection(context_edge_set)
        
        # Remove path_and_context_edges from other categories to avoid double drawing
        context_only_edges = context_edge_set - path_and_context_edges
        reasoning_only_edges = reasoning_edges - path_and_context_edges

        # Draw edges with different styles
        all_edges = list(graph.edges())

        # Other edges (background)
        other_edges = [(u, v) for u, v in all_edges
                      if (u, v) not in context_edge_set and (u, v) not in reasoning_edges]
        if other_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=other_edges,
                                 edge_color=self.colors['other_edge'],
                                 width=0.5 * edge_width_scale, alpha=0.3, ax=ax)

        # Context-only edges (red)
        context_only_edges_in_graph = [(u, v) for u, v in context_only_edges if graph.has_edge(u, v)]
        if context_only_edges_in_graph:
            nx.draw_networkx_edges(graph, pos, edgelist=context_only_edges_in_graph,
                                 edge_color=self.colors['context_edge'],
                                 width=2 * edge_width_scale, alpha=0.8, ax=ax)

        # Reasoning-only edges (teal)
        reasoning_only_edges_in_graph = [(u, v) for u, v in reasoning_only_edges if graph.has_edge(u, v)]
        if reasoning_only_edges_in_graph:
            nx.draw_networkx_edges(graph, pos, edgelist=reasoning_only_edges_in_graph,
                                 edge_color=self.colors['reasoning_path'],
                                 width=3 * edge_width_scale, alpha=1.0, ax=ax,
                                 style='solid')

        # Edges that are both on path and in context (purple)
        path_and_context_edges_in_graph = [(u, v) for u, v in path_and_context_edges if graph.has_edge(u, v)]
        if path_and_context_edges_in_graph:
            nx.draw_networkx_edges(graph, pos, edgelist=path_and_context_edges_in_graph,
                                 edge_color=self.colors['path_and_context'],
                                 width=4 * edge_width_scale, alpha=1.0, ax=ax,
                                 style='solid')

        # Categorize and draw nodes
        nodes = list(graph.nodes())
        node_colors = []
        node_sizes = []

        for node in nodes:
            base_size = 300 * node_size_scale

            if node == source_idx:
                node_colors.append(self.colors['source_node'])
                node_sizes.append(base_size * 2)
            elif node == reachable_idx:
                node_colors.append(self.colors['reachable_target'])
                node_sizes.append(base_size * 2)
            elif node == unreachable_idx:
                node_colors.append(self.colors['unreachable_target'])
                node_sizes.append(base_size * 2)
            elif node in path_indices:
                node_colors.append(self.colors['path_node'])
                node_sizes.append(base_size * 1.5)
            else:
                node_colors.append(self.colors['other_node'])
                node_sizes.append(base_size)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                             node_size=node_sizes, alpha=0.9, ax=ax)

        # Add labels if requested
        if show_labels:
            labels = {node: self.concepts[node] for node in nodes}
            nx.draw_networkx_labels(graph, pos, labels, font_size=label_font_size, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    def _plot_sample_info(self, ax, sample_data):
        """Plot sample information and legend."""
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, 'Sample Information', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=ax.transAxes)

        # Question
        question_text = f"Question: {sample_data['question']}"
        ax.text(0.05, 0.85, question_text, ha='left', va='top',
                fontsize=12, fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))

        # Answer
        answer_text = f"Answer: {sample_data['answer']}"
        ax.text(0.05, 0.75, answer_text, ha='left', va='top',
                fontsize=12, fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))

        # Reasoning path
        path_text = f"Reasoning Path:\n{' → '.join(sample_data['path'])}"
        ax.text(0.05, 0.65, path_text, ha='left', va='top',
                fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.5))

        # Legend
        legend_y = 0.45
        ax.text(0.05, legend_y, 'Legend:', ha='left', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)

        legend_items = [
            ('Source Node', self.colors['source_node']),
            ('Reachable Target', self.colors['reachable_target']),
            ('Unreachable Target', self.colors['unreachable_target']),
            ('Path Nodes', self.colors['path_node']),
            ('Context Edges', self.colors['context_edge']),
            ('Reasoning Path', self.colors['reasoning_path']),
            ('Path + Context', self.colors['path_and_context']),
        ]

        y_offset = 0.03
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y - 0.04 - (i * y_offset)

            # Draw colored square
            rect = FancyBboxPatch((0.05, y_pos - 0.01), 0.03, 0.02,
                                boxstyle="round,pad=0.002",
                                facecolor=color, alpha=0.8,
                                transform=ax.transAxes)
            ax.add_patch(rect)

            # Add label
            ax.text(0.1, y_pos, label, ha='left', va='center',
                    fontsize=10, transform=ax.transAxes)

        # Statistics
        stats_y = legend_y - 0.25
        ax.text(0.05, stats_y, 'Statistics:', ha='left', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)

        stats_text = f"""Path Length: {sample_data['path_length']}
Source: {sample_data['source']}
Reachable: {sample_data['reachable_target']}
Unreachable: {sample_data['unreachable_target']}"""

        ax.text(0.05, stats_y - 0.04, stats_text, ha='left', va='top',
                fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))

    def visualize_multiple_samples(self, num_samples=4, figsize=(20, 16),
                                 save_path=None):
        """Visualize multiple samples in a grid."""
        rows = int(np.ceil(num_samples / 2))
        fig, axes = plt.subplots(rows, 2, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            row, col = i // 2, i % 2

            sample = self.generator.create_sample()
            if sample is None:
                axes[row, col].text(0.5, 0.5, "Failed to generate sample",
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
                continue

            # Create mini visualization
            source_idx = self.concepts.index(sample['source'])
            reachable_idx = self.concepts.index(sample['reachable_target'])
            unreachable_idx = self.concepts.index(sample['unreachable_target'])
            path_indices = [self.concepts.index(concept) for concept in sample['path']]
            context_edges = self._extract_context_edges(sample['context'])

            # Get relevant nodes for subgraph
            relevant_nodes = set(path_indices + [unreachable_idx])
            for u, v in context_edges[:10]:  # Limit context edges for clarity
                relevant_nodes.update([u, v])

            if len(relevant_nodes) > 30:
                # Keep most important nodes
                important = set(path_indices + [unreachable_idx])
                others = list(relevant_nodes - important)
                others = set(random.sample(others, min(20, len(others))))
                relevant_nodes = important | others

            subgraph = self.graph.subgraph(relevant_nodes)

            self._plot_graph(subgraph, axes[row, col], source_idx, reachable_idx,
                           unreachable_idx, path_indices, context_edges[:10],
                           'spring', 0.7, False, 8, 0.7,
                           f"Sample {i+1}: {sample['source']} → {sample['reachable_target']}")

        # Remove empty subplots
        for i in range(num_samples, rows * 2):
            row, col = i // 2, i % 2
            fig.delaxes(axes[row, col])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig


# Example usage
if __name__ == "__main__":
    from generate_shared_graph_cot_2 import SubProsQADataset
    
    dataset = SubProsQADataset(
        num_nodes=100,
        num_edges=300,
        num_nodes_precompute=100,
        num_samples=100,
        num_context_edges=25,
        representation='structured',  # or 'natural' or 'hybrid'
        context_edge_proximity_weight=100.0,
        candidate_samples_path='candidate_samples.json',
        load_candidate_samples=False,
        dataset_path='prosqa_dataset.json',
        load_dataset=False,
        depth_range=(3, 6),
        seed=42
    )

    sample = dataset.dataset[0]
    visualizer = ProsQAGraphVisualizer(dataset)

    if sample:
        print("Sample generated successfully!")
        print(f"Question: {sample['question']}")
        print(f"Path: {' -> '.join(sample['path'])}")

        # Visualize the sample
        fig = visualizer.visualize_sample(
            sample_data=sample,
            figsize=(18, 10),
            layout='spring',
            node_size_scale=1.2,
            show_labels=False,
            save_path='prosqa_sample.png'
        )

        # Visualize multiple samples
        print("\nGenerating multiple samples...")
        fig_multi = visualizer.visualize_multiple_samples(
            num_samples=6,
            figsize=(24, 18),
            save_path='prosqa_multiple_samples.png'
        )

    else:
        print("Failed to generate sample")