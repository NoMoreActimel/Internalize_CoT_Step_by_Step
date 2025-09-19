import argparse
import json
import networkx as nx
import os
import random

from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

class SubProsQADataset:
    """
    Generate ProsQA-style dataset with binary questions with a single huge graph shared between all samples.
    Question format: "Is X a Y or a Z?" in the symbolic format, one node is reachable, another is not.
    Context format: [EDGES] X->Y, Y->Z, Z->X;
    where edges are randomly sampled from the graph with higher probability for edges around the path.
    """
    
    def __init__(
        self,
        num_nodes=1000,
        num_edges=2000,
        num_samples=10000,
        num_context_edges=20,
        representation='structured',
        context_edge_proximity_weight=5.0,
        num_nodes_precompute=None, # for all nodes by default
        candidate_samples_path=None, # for precomputed samples, either save or load
        load_candidate_samples=False,
        dataset_path=None, # for precomputed dataset, either save or load
        load_dataset=False,
        depth_range=(3, 6),
        seed=42
    ):
        random.seed(seed)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_samples = num_samples
        self.num_context_edges = num_context_edges
        self.representation = representation
        self.context_edge_proximity_weight = context_edge_proximity_weight
        self.num_nodes_precompute = num_nodes_precompute
        self.candidate_samples_path = candidate_samples_path
        self.load_candidate_samples = load_candidate_samples
        self.dataset_path = dataset_path
        self.load_dataset = load_dataset
        self.depth_range = depth_range

        self.concepts = [f"concept{i:04d}" for i in range(num_nodes)]
        self.graph = self._build_dag()
        self.all_edges = list(self.graph.edges())

        if self.load_dataset and self.dataset_path:
            print(f"Loading dataset from {self.dataset_path}...")
            self.dataset = self._load_dataset(self.dataset_path)
            self.candidate_samples = None
        else:
            # Generate new dataset
            if self.load_candidate_samples and self.candidate_samples_path:
                self.candidate_samples = self._load_candidate_samples(self.candidate_samples_path)
            else:
                self.candidate_samples = self._precompute_candidate_samples()
                if self.candidate_samples_path:
                    self._save_candidate_samples(self.candidate_samples, self.candidate_samples_path)
            
            self.dataset = self.create_dataset()
            self._save_dataset(self.dataset, self.dataset_path)

        print(f"Graph created: {num_nodes} nodes, {self.graph.number_of_edges()} edges")
        self._print_graph_stats()
    
    def _build_dag(self):
        """Build a sparse DAG suitable for reasoning."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))
        
        edges_added = 0
        attempts = 0
        max_attempts = self.num_edges * 20
        
        while edges_added < self.num_edges and attempts < max_attempts:
            attempts += 1
            u = random.randint(0, self.num_nodes - 2)
            max_v = min(self.num_nodes - 1, u + random.randint(1, self.num_nodes // 5))  # some kind of edge structure
            v = random.randint(u + 1, max_v)
            
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                edges_added += 1
        
        return G
    
    def _precompute_candidate_samples(self):
        n = self.num_nodes if self.num_nodes_precompute is None else self.num_nodes_precompute
        min_depth, max_depth = self.depth_range
        sampled_sources = random.sample(list(range(self.num_nodes)), n)

        candidate_samples = defaultdict(dict)

        for source in sampled_sources:
            reachable_paths = nx.single_source_shortest_path(self.graph, source, cutoff=max_depth)
            candidate_samples[source] = {}
            candidate_samples[source]["valid_targets"] = set()
            candidate_samples[source]["invalid_targets"] = set()
            candidate_samples[source]["valid_paths"] = {}

            for node, path in reachable_paths.items():
                if len(path) - 1 >= min_depth and len(path) - 1 <= max_depth:
                    candidate_samples[source]["valid_targets"].add(node)
                    candidate_samples[source]["valid_paths"][node] = {"path": path, "path_length": len(path) - 1}
            
            reachable_nodes = set(reachable_paths.keys())
            for node in range(self.num_nodes):
                if node not in reachable_nodes: # candidate_samples[source]["valid_targets"]:
                    candidate_samples[source]["invalid_targets"].add(node)

            # in case we want to include depth out of [min_depth, max_depth] into invalid targets, (by default not)
            # invalid_targets = [node for node in range(self.num_nodes) if node not in valid_targets]

            assert len(candidate_samples[source]["invalid_targets"]), "Found zero unreachable targets with specified depth"
        
        print(f"Generated candidate samples with target paths for {n} / {self.num_nodes} random nodes in graph!")
        return candidate_samples
    
    @staticmethod
    def _save_candidate_samples(candidate_samples: Dict, candidate_samples_path: str):
        # Convert sets to lists for JSON serialization
        serializable_samples = {}
        for source, data in candidate_samples.items():
            serializable_samples[source] = {}
            for key, value in data.items():
                if isinstance(value, set):
                    serializable_samples[source][key] = list(value)
                else:
                    serializable_samples[source][key] = value
        
        with open(candidate_samples_path, "w") as f:
            json.dump(serializable_samples, f)
        
    @staticmethod
    def _load_candidate_samples(candidate_samples_path: str):
        with open(candidate_samples_path, "r") as f:
            loaded_samples = json.load(f)
        
        # Convert lists back to sets after loading
        for source, data in loaded_samples.items():
            for key, value in data.items():
                if key in ['valid_targets', 'invalid_targets'] and isinstance(value, list):
                    loaded_samples[source][key] = set(value)
        
        return loaded_samples

    @staticmethod
    def _save_dataset(dataset: List[Dict], dataset_path: str):
        """Save the entire dataset to JSON file."""
        print(f"Saving dataset to {dataset_path}...")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved with {len(dataset)} samples!")
        
    @staticmethod
    def _load_dataset(dataset_path: str) -> List[Dict]:
        """Load the entire dataset from JSON file."""
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        print(f"Dataset loaded with {len(dataset)} samples!")
        return dataset

    # def _find_reachable_and_unreachable(self, source: int, depth_range: Tuple[int, int]) -> Optional[Dict]:
        """
        Find a reachable target within depth range and an unreachable target.
        """
        min_depth, max_depth = depth_range
        
        # Use BFS with depth limit (very efficient for depth <= 10)
        
        # Find targets within the depth range
        
        if not valid_targets:
            return None
        
        # Choose a reachable target
        reachable_target = random.choice(valid_targets)
        path_length = reachable[reachable_target]
        
        # Find the actual path
        path = nx.shortest_path(self.graph, source, reachable_target)
        
        # Find an unreachable node (or one that's too far)
        unreachable_candidates = []
        
        # Option 1: Nodes that are completely unreachable
        all_reachable = set(nx.single_source_shortest_path_length(
            self.graph, source, cutoff=max_depth + 5
        ).keys())
        completely_unreachable = set(range(self.num_nodes)) - all_reachable
        
        if completely_unreachable:
            unreachable_candidates.extend(list(completely_unreachable))
        
        # Option 2: Nodes that are reachable but outside depth range
        too_far = [
            node for node, dist in reachable.items()
            if dist > max_depth
        ]
        unreachable_candidates.extend(too_far)
        
        # Option 3: If graph is too connected, use any other node
        if not unreachable_candidates:
            unreachable_candidates = [n for n in range(self.num_nodes) 
                                     if n != source and n != reachable_target]
        
        if not unreachable_candidates:
            return None
        
        unreachable_target = random.choice(unreachable_candidates)
        
        return {
            'source': source,
            'reachable_target': reachable_target,
            'unreachable_target': unreachable_target,
            'path': path,
            'path_length': path_length
        }
    
    def _sample_context_edges(self, path: List[int], num_edges: int, proximity_weight: float = 3.0) -> List[Tuple[int, int]]:
        """
        Sample edges for context, biased toward the solution path.
        """
        path_nodes = set(path)
        
        # Separate path-related and other edges for efficient sampling
        path_related_edges = []
        other_edges = []
        
        for i, (u, v) in enumerate(self.all_edges):
            if u in path_nodes or v in path_nodes:
                path_related_edges.append(i)
            else:
                other_edges.append(i)
        
        num_to_sample = min(num_edges, len(self.all_edges))
        
        # Calculate how many edges to sample from each category based on weights
        total_path_weight = len(path_related_edges) * proximity_weight
        total_other_weight = len(other_edges) * 1.0
        total_weight = total_path_weight + total_other_weight
        
        if total_weight == 0:
            return []
            
        # Proportion of samples from path-related edges
        path_proportion = total_path_weight / total_weight
        num_path_samples = min(int(num_to_sample * path_proportion), len(path_related_edges))
        num_other_samples = min(num_to_sample - num_path_samples, len(other_edges))
        
        # Sample from each category
        sampled_indices = []
        if num_path_samples > 0 and path_related_edges:
            sampled_indices.extend(random.choices(path_related_edges, k=num_path_samples))
        if num_other_samples > 0 and other_edges:
            sampled_indices.extend(random.choices(other_edges, k=num_other_samples))
        
        # Fill remaining slots if needed
        remaining = num_to_sample - len(sampled_indices)
        if remaining > 0:
            all_available = path_related_edges + other_edges
            available_indices = [i for i in all_available if i not in sampled_indices]
            if available_indices:
                additional = random.choices(available_indices, k=min(remaining, len(available_indices)))
                sampled_indices.extend(additional)
        
        return [self.all_edges[i] for i in sampled_indices[:num_to_sample]]
    
    def create_sample(self, num_context_edges=20, representation='structured') -> Optional[Dict]:
        """
        Args:
            num_context_edges: number of edges to include in context
            representation: 'structured', 'natural', or 'hybrid'
        """
        # Try to find a valid source with both reachable and unreachable targets
        max_attempts = 100
        
        for _ in range(max_attempts):
            # source = random.randint(0, self.num_nodes - 1)
            # result = self._find_reachable_and_unreachable(source, depth_range)
            source = random.choice(list(self.candidate_samples.keys()))
            result = self.candidate_samples[source]
            if len(result['valid_targets']) > 0 and len(result['invalid_targets']) > 0:
                break
        else:
            return None
        
        # Extract components
        source_concept = self.concepts[source]
        reachable_target = random.choice(list(result['valid_targets']))
        unreachable_target = random.choice(list(result['invalid_targets']))
        
        reachable_concept = self.concepts[reachable_target]
        unreachable_concept = self.concepts[unreachable_target]
        path = result['valid_paths'][reachable_target]['path']
        path_length = result['valid_paths'][reachable_target]['path_length']
        
        # Randomly order the options in the question
        if random.random() < 0.5:
            question = f"{source_concept} is {reachable_concept} or {unreachable_concept}"
            answer = reachable_concept
            correct_option = 0
        else:
            question = f"{source_concept} is {unreachable_concept} or {reachable_concept}"
            answer = reachable_concept
            correct_option = 1
        
        # Sample context edges
        context_edges = self._sample_context_edges(path, num_context_edges, proximity_weight=self.context_edge_proximity_weight)
        
        # Generate representation based on type
        if representation == 'structured':
            context = self._structured_context(context_edges)
            reasoning = self._structured_reasoning(path)
        elif representation == 'natural':
            context = self._natural_context(context_edges)
            reasoning = self._natural_reasoning(path)
        else:  # hybrid
            context = self._hybrid_context(context_edges)
            reasoning = self._hybrid_reasoning(path)
        
        return {
            'question': question,
            'context': context,
            'reasoning_steps': reasoning,
            'answer': f"{source_concept} is {answer}",
            'correct_option': correct_option,
            'path': [self.concepts[n] for n in path],
            'path_length': path_length,
            'source': source_concept,
            'reachable_target': reachable_concept,
            'unreachable_target': unreachable_concept
        }
    
    def _structured_context(self, edges: List[Tuple[int, int]]) -> str:
        """Generate structured context representation."""
        edge_strs = [f"{self.concepts[u]} -> {self.concepts[v]}" for u, v in edges]
        return f"[EDGES] {', '.join(edge_strs)}"
    
    def _natural_context(self, edges: List[Tuple[int, int]]) -> str:
        """Generate natural language context."""
        statements = []
        for u, v in edges:
            templates = [
                f"Every {self.concepts[u]} is a {self.concepts[v]}.",
                f"All {self.concepts[u]}s are {self.concepts[v]}s.",
                f"{self.concepts[u].capitalize()}s are {self.concepts[v]}s."
            ]
            statements.append(random.choice(templates))
        random.shuffle(statements)
        return ' '.join(statements)
    
    def _hybrid_context(self, edges: List[Tuple[int, int]]) -> str:
        """Generate hybrid context."""
        structured = [f"{self.concepts[u]}->{self.concepts[v]}" for u, v in edges[:len(edges)//2]]
        natural = []
        for u, v in edges[len(edges)//2:]:
            natural.append(f"Every {self.concepts[u]} is a {self.concepts[v]}.")
        
        return f"[EDGES] {' '.join(structured)} [FACTS] {' '.join(natural)}"
    
    def _structured_reasoning(self, path: List[int]) -> List[str]:
        """Generate structured reasoning steps."""
        steps = []
        for i in range(len(path) - 1):
            steps.append(f"{self.concepts[path[i]]}->{self.concepts[path[i+1]]}")
        return steps
    
    def _natural_reasoning(self, path: List[int]) -> List[str]:
        """Generate natural language reasoning."""
        steps = []
        for i in range(len(path) - 1):
            steps.append(f"{self.concepts[path[i]]} is a {self.concepts[path[i+1]]}")
        return steps
    
    def _hybrid_reasoning(self, path: List[int]) -> List[str]:
        """Generate hybrid reasoning."""
        steps = []
        for i in range(len(path) - 1):
            if i % 2 == 0:
                steps.append(f"{self.concepts[path[i]]}->{self.concepts[path[i+1]]}")
            else:
                steps.append(f"{self.concepts[path[i]]} is a {self.concepts[path[i+1]]}")
        return steps
    
    def create_dataset(self):
        dataset = []
        failed = 0
        
        pbar_update = max(1, self.num_samples // 20)
        
        while len(dataset) < self.num_samples and failed < self.num_samples * 2:
            if len(dataset) % pbar_update == 0:
                print(f"Created {len(dataset)}/{self.num_samples} samples...")
            
            sample = self.create_sample(self.num_context_edges, self.representation)
            if sample:
                dataset.append(sample)
            else:
                failed += 1
        
        print(f"Successfully created dataset with {len(dataset)} samples!")
        
        self._print_dataset_stats(dataset)
        return dataset
    
    def _print_graph_stats(self):
        """Print graph statistics."""
        print(f"Graph statistics:")
        print(f"  Number of nodes: {self.num_nodes}")
        print(f"  Number of edges: {self.num_edges}")
        # print(f"  Number of components: {nx.number_connected_components(self.graph)}")
        # print(f"  Diameter: {nx.diameter(self.graph)}")
        print(f"  Average clustering coefficient: {nx.average_clustering(self.graph)}")
        # print(f"  Average shortest path length: {nx.average_shortest_path_length(self.graph)}")
        print(f"  Degree distribution: {nx.degree_histogram(self.graph)}")
        # print(f"  Connected components: {nx.connected_components(self.graph)}")
        print(f"  Transitivity: {nx.transitivity(self.graph)}")
        print(f"  Closeness centrality: {nx.closeness_centrality(self.graph)}")
        print(f"  Betweenness centrality: {nx.betweenness_centrality(self.graph)}")
        # print(f"  Eigenvector centrality: {nx.eigenvector_centrality(self.graph)}")
        print(f"  PageRank: {nx.pagerank(self.graph)}")
        print(f"  Clustering coefficient: {nx.clustering(self.graph)}")
        print(f"  Degree centrality: {nx.degree_centrality(self.graph)}")
        print(f"  Degree assortativity: {nx.degree_assortativity_coefficient(self.graph)}")
    
    def _print_dataset_stats(self, dataset):
        """Compute dataset statistics."""
        path_lengths = [s['path_length'] for s in dataset]
        stats = {
            'num_samples': len(dataset),
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'min_path_length': min(path_lengths) if path_lengths else 0,
            'max_path_length': max(path_lengths) if path_lengths else 0,
        }
        print(f"Dataset statistics:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Generate SubProsQA dataset with shared graph CoT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num_nodes', type=int, default=1000,
                        help='Number of nodes in the graph')
    parser.add_argument('--num_edges', type=int, default=3000,
                        help='Number of edges in the graph')
    parser.add_argument('--num_nodes_precompute', type=int, default=1000,
                        help='Number of nodes to precompute (default: same as num_nodes)')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--num_context_edges', type=int, default=25,
                        help='Number of context edges per sample')
    parser.add_argument('--depth_range', type=int, nargs=2, default=[3, 10],
                        help='Min and max depth for path generation (e.g., --depth_range 3 10)')
    
    parser.add_argument('--representation', type=str, default='structured',
                        choices=['structured', 'natural', 'hybrid'],
                        help='Type of representation to use')
    parser.add_argument('--context_edge_proximity_weight', type=float, default=5.0,
                        help='Weight for proximity-based edge selection')
    
    parser.add_argument('--candidate_samples_path', type=str, default='candidate_samples.json',
                        help='Path to save/load candidate samples')
    parser.add_argument('--dataset_path', type=str, default='prosqa_dataset.json',
                        help='Path to save/load dataset')
    parser.add_argument('--config_path', type=str, default='config.json',
                        help='Path to save configuration file')
    parser.add_argument('--load_candidate_samples', action='store_true',
                        help='Load precomputed candidate samples')
    parser.add_argument('--load_dataset', action='store_true',
                        help='Load precomputed dataset')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--show_samples', type=int, default=5,
                        help='Number of sample outputs to display')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress sample output display')
    
    args = parser.parse_args()
    
    if args.candidate_samples_path:
        os.makedirs(os.path.dirname(args.candidate_samples_path) or '.', exist_ok=True)
    if args.dataset_path:
        os.makedirs(os.path.dirname(args.dataset_path) or '.', exist_ok=True)
    
    print("="*60)
    print("SubProsQA Dataset Generation")
    print("="*60)
    print(f"Graph: {args.num_nodes} nodes, {args.num_edges} edges")
    print(f"Samples: {args.num_samples}")
    print(f"Context edges per sample: {args.num_context_edges}")
    print(f"Representation: {args.representation}")
    print(f"Depth range: {args.depth_range[0]}-{args.depth_range[1]}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    prosqa_generator = SubProsQADataset(
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        num_nodes_precompute=args.num_nodes_precompute,
        num_samples=args.num_samples,
        num_context_edges=args.num_context_edges,
        representation=args.representation,
        context_edge_proximity_weight=args.context_edge_proximity_weight,
        candidate_samples_path=args.candidate_samples_path,
        load_candidate_samples=args.load_candidate_samples,
        dataset_path=args.dataset_path,
        load_dataset=args.load_dataset,
        depth_range=tuple(args.depth_range),
        seed=args.seed
    )

    dataset = prosqa_generator.dataset
    
    if not args.quiet:
        random_indices = random.sample(range(len(dataset)), min(args.show_samples, len(dataset)))
        
        for i, random_index in enumerate(random_indices):
            print(f"\n{'='*60}")
            print(f"SAMPLE OUTPUT {i+1}/{len(random_indices)}")
            print("="*60)
            
            sample = dataset[random_index]
            print(f"\nQuestion: {sample['question']}")
            print("Context:")
            for line in [sample['context'][i:i+200] for i in range(0, len(sample['context']), 200)]:
                print(line)
            
            print(f"Reasoning path: {' -> '.join(sample['path'])}")
            print(f"Answer: {sample['answer']}")
            print(f"\nPath from {sample['source']} to {sample['reachable_target']}: YES (length {sample['path_length']})")
            print(f"Path from {sample['source']} to {sample['unreachable_target']}: NO")
    
    # Save configuration
    config = {
        'num_nodes': args.num_nodes,
        'num_edges': args.num_edges,
        'num_nodes_precompute': args.num_nodes_precompute,
        'num_samples': args.num_samples,
        'num_context_edges': args.num_context_edges,
        'representation': args.representation,
        'context_edge_proximity_weight': args.context_edge_proximity_weight,
        'depth_range': args.depth_range,
        'seed': args.seed,
        'candidate_samples_path': args.candidate_samples_path,
        'dataset_path': args.dataset_path,
        'config_path': args.config_path,
        'load_candidate_samples': args.load_candidate_samples,
        'load_dataset': args.load_dataset,
        'show_samples': args.show_samples,
        'total_samples_generated': len(dataset)
    }
    
    with open(args.config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(dataset)}")
    if args.dataset_path:
        print(f"Dataset saved to: {args.dataset_path}")
    print(f"Configuration saved to: {args.config_path}")
