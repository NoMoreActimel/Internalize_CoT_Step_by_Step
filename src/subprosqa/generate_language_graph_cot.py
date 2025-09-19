import random
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Set
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

class LanguageSubProsQADataset:
    """
    Generate ProsQA-style dataset with binary questions using a language-based graph.
    The graph is built from C4 dataset token transitions, where:
    - Nodes = tokens in vocabulary
    - Edges = transitions from token n to token n+1 (weighted by frequency)
    - Questions ask about reachability in this language graph
    """
    
    def __init__(
        self,
        vocab_size=10000,  # Limit vocabulary size for manageable graphs
        num_samples=10000,
        num_context_edges=20,
        representation='structured',
        context_edge_proximity_weight=5.0,
        num_nodes_precompute=None,
        candidate_samples_path=None,
        load_candidate_samples=False,
        dataset_path=None,
        load_dataset=False,
        depth_range=(3, 6),
        c4_subset_size=10000,  # Number of C4 samples to process
        tokenizer_name="meta-llama/Llama-3.2-1B",  # or use a smaller one
        seed=42
    ):
        random.seed(seed)
        self.vocab_size = vocab_size
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
        self.c4_subset_size = c4_subset_size
        self.tokenizer_name = tokenizer_name
        
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Build language graph from C4 dataset
        print("Building language graph from C4 dataset...")
        self.graph = self._build_language_dag()
        self.all_edges = list(self.graph.edges(data=True))  # Include edge data (weights)
        
        # Create token concepts (first vocab_size tokens)
        self.concepts = [f"token_{i:05d}" for i in range(min(vocab_size, len(self.tokenizer.get_vocab())))]
        
        # Handle dataset loading/creation
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

        print(f"Language graph created: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _build_language_dag(self):
        """Build a weighted DAG from C4 dataset token transitions."""
        print("Loading C4 dataset...")
        
        # Load a subset of C4 dataset
        try:
            dataset = load_dataset("c4", "en", split="train", streaming=True)
        except:
            print("Failed to load C4 dataset, using a smaller alternative...")
            # Fallback to a smaller dataset if C4 is not available
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Initialize weighted directed graph
        G = nx.DiGraph()
        
        # Track token frequencies for vocabulary selection
        token_counts = Counter()
        edge_weights = defaultdict(int)
        
        print(f"Processing {self.c4_subset_size} samples from C4 dataset...")
        
        # Process samples to build token transition graph
        sample_count = 0
        for sample in tqdm(dataset, desc="Processing C4 samples"):
            if sample_count >= self.c4_subset_size:
                break
                
            text = sample.get('text', '')
            if not text or len(text.strip()) < 10:  # Skip very short texts
                continue
            
            # Tokenize the text
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
            except:
                continue
            
            if len(tokens) < 2:  # Need at least 2 tokens for transitions
                continue
            
            # Count token frequencies
            for token in tokens:
                token_counts[token] += 1
            
            # Build token transitions (edges)
            for i in range(len(tokens) - 1):
                from_token = tokens[i]
                to_token = tokens[i + 1]
                edge_weights[(from_token, to_token)] += 1
            
            sample_count += 1
        
        print(f"Processed {sample_count} samples")
        print(f"Found {len(token_counts)} unique tokens")
        print(f"Found {len(edge_weights)} unique transitions")
        
        # Select top vocab_size most frequent tokens
        top_tokens = [token for token, count in token_counts.most_common(self.vocab_size)]
        token_to_idx = {token: idx for idx, token in enumerate(top_tokens)}
        
        print(f"Selected top {len(top_tokens)} tokens for vocabulary")
        
        # Build the graph with selected tokens only
        for (from_token, to_token), weight in edge_weights.items():
            if from_token in token_to_idx and to_token in token_to_idx:
                from_idx = token_to_idx[from_token]
                to_idx = token_to_idx[to_token]
                
                if G.has_edge(from_idx, to_idx):
                    # Add to existing edge weight
                    G[from_idx][to_idx]['weight'] += weight
                else:
                    # Create new edge with weight
                    G.add_edge(from_idx, to_idx, weight=weight)
        
        # Store token mapping for later use
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        
        print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Print some statistics
        weights = [data['weight'] for u, v, data in G.edges(data=True)]
        if weights:
            print(f"Edge weight statistics:")
            print(f"  Min weight: {min(weights)}")
            print(f"  Max weight: {max(weights)}")
            print(f"  Mean weight: {np.mean(weights):.2f}")
            print(f"  Median weight: {np.median(weights):.2f}")
        
        return G
    
    def _precompute_candidate_samples(self):
        """Precompute candidate samples for efficient dataset generation."""
        n = self.graph.number_of_nodes() if self.num_nodes_precompute is None else self.num_nodes_precompute
        min_depth, max_depth = self.depth_range
        
        # Sample random source nodes
        all_nodes = list(self.graph.nodes())
        sampled_sources = random.sample(all_nodes, min(n, len(all_nodes)))
        
        candidate_samples = defaultdict(dict)
        
        print(f"Precomputing candidate samples for {len(sampled_sources)} sources...")
        
        for source in tqdm(sampled_sources, desc="Computing reachable paths"):
            # Use weighted shortest paths for more realistic reasoning
            try:
                reachable_paths = nx.single_source_shortest_path(self.graph, source, cutoff=max_depth)
            except:
                continue
                
            candidate_samples[source] = {}
            candidate_samples[source]["valid_targets"] = set()
            candidate_samples[source]["invalid_targets"] = set()
            candidate_samples[source]["valid_paths"] = {}
            
            for node, path in reachable_paths.items():
                path_length = len(path) - 1
                if min_depth <= path_length <= max_depth:
                    candidate_samples[source]["valid_targets"].add(node)
                    candidate_samples[source]["valid_paths"][node] = {
                        "path": path, 
                        "path_length": path_length,
                        "path_weight": self._calculate_path_weight(path)
                    }
            
            # Find unreachable targets
            reachable_nodes = set(reachable_paths.keys())
            for node in all_nodes:
                if node not in reachable_nodes:
                    candidate_samples[source]["invalid_targets"].add(node)
            
            if not candidate_samples[source]["invalid_targets"]:
                # If no unreachable targets, use nodes that are too far
                too_far_nodes = [node for node, path in reachable_paths.items() 
                               if len(path) - 1 > max_depth]
                candidate_samples[source]["invalid_targets"] = set(too_far_nodes[:10])  # Limit to 10
        
        print(f"Generated candidate samples for {len(candidate_samples)} sources!")
        return candidate_samples
    
    def _calculate_path_weight(self, path):
        """Calculate total weight of a path in the weighted graph."""
        total_weight = 0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i + 1]):
                total_weight += self.graph[path[i]][path[i + 1]].get('weight', 1)
        return total_weight
    
    @staticmethod
    def _save_candidate_samples(candidate_samples: Dict, candidate_samples_path: str):
        """Save candidate samples to JSON file."""
        if not candidate_samples_path:
            return
            
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
        """Load candidate samples from JSON file."""
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
        if not dataset_path:
            return
            
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
    
    def _sample_context_edges(self, path: List[int], num_edges: int, proximity_weight: float = 3.0) -> List[Tuple[int, int]]:
        """
        Sample edges for context, biased toward the solution path.
        Adapted for weighted graphs.
        """
        path_nodes = set(path)
        
        # Separate path-related and other edges for efficient sampling
        path_related_edges = []
        other_edges = []
        
        for i, (u, v, data) in enumerate(self.all_edges):
            if u in path_nodes or v in path_nodes:
                path_related_edges.append((i, data.get('weight', 1)))
            else:
                other_edges.append((i, data.get('weight', 1)))
        
        num_to_sample = min(num_edges, len(self.all_edges))
        
        # Calculate sampling probabilities based on edge weights
        path_weights = [weight for _, weight in path_related_edges]
        other_weights = [weight for _, weight in other_edges]
        
        total_path_weight = sum(weight * proximity_weight for weight in path_weights)
        total_other_weight = sum(other_weights)
        total_weight = total_path_weight + total_other_weight
        
        if total_weight == 0:
            return []
        
        # Sample edges based on weights
        sampled_indices = []
        
        # Sample from path-related edges
        if path_related_edges and total_path_weight > 0:
            path_probs = [weight * proximity_weight / total_path_weight for _, weight in path_related_edges]
            path_indices = [idx for idx, _ in path_related_edges]
            num_path_samples = min(int(num_to_sample * total_path_weight / total_weight), len(path_related_edges))
            if num_path_samples > 0:
                sampled_indices.extend(random.choices(path_indices, weights=path_probs, k=num_path_samples))
        
        # Sample from other edges
        if other_edges and len(sampled_indices) < num_to_sample:
            other_probs = [weight / total_other_weight for _, weight in other_edges]
            other_indices = [idx for idx, _ in other_edges]
            remaining = num_to_sample - len(sampled_indices)
            if remaining > 0:
                sampled_indices.extend(random.choices(other_indices, weights=other_probs, k=remaining))
        
        return [self.all_edges[i][:2] for i in sampled_indices[:num_to_sample]]
    
    def create_sample(self, num_context_edges=20, representation='structured') -> Optional[Dict]:
        """Create a single sample for the dataset."""
        max_attempts = 100
        
        for _ in range(max_attempts):
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
        path_weight = result['valid_paths'][reachable_target]['path_weight']
        
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
            'path_weight': path_weight,
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
        """Create the complete dataset."""
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
    
    def _print_dataset_stats(self, dataset):
        """Print dataset statistics."""
        path_lengths = [s['path_length'] for s in dataset]
        path_weights = [s['path_weight'] for s in dataset]
        
        stats = {
            'num_samples': len(dataset),
            'avg_path_length': sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            'min_path_length': min(path_lengths) if path_lengths else 0,
            'max_path_length': max(path_lengths) if path_lengths else 0,
            'avg_path_weight': sum(path_weights) / len(path_weights) if path_weights else 0,
            'min_path_weight': min(path_weights) if path_weights else 0,
            'max_path_weight': max(path_weights) if path_weights else 0,
        }
        print(f"Dataset statistics:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value}")


# Example usage
if __name__ == "__main__":
    # Create language-based dataset
    dataset = LanguageSubProsQADataset(
        vocab_size=5000,  # Smaller vocab for faster processing
        num_samples=1000,
        num_context_edges=25,
        representation='structured',
        context_edge_proximity_weight=5.0,
        candidate_samples_path='language_candidate_samples.json',
        load_candidate_samples=False,
        dataset_path='language_prosqa_dataset.json',
        load_dataset=False,
        depth_range=(3, 8),
        c4_subset_size=5000,  # Process 5k C4 samples
        tokenizer_name="meta-llama/Llama-3.2-1B",
        seed=42
    )

    # Print some sample outputs
    if dataset.dataset:
        print("\n" + "="*60)
        print("SAMPLE OUTPUTS")
        print("="*60)
        
        for i, sample in enumerate(dataset.dataset[:3]):
            print(f"\nSample {i+1}:")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print(f"Path: {' -> '.join(sample['path'])}")
            print(f"Path length: {sample['path_length']}")
            print(f"Path weight: {sample['path_weight']}")
            print(f"Context: {sample['context'][:200]}...")
