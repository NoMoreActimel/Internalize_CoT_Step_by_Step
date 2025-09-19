"""
Generate CoT reasoning dataset with shared implicit graph structure.

Unlike ProsQA/ProntoQA which generate a new graph per sample, this creates 
a single large graph shared among all samples. The graph structure is implicit
(not provided in context) and reasoning traces of controlled depth are sampled.

Context edges are sampled with higher probability for edges close to the reasoning path.
"""

import argparse
import json
import random
import os
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


class SharedGraphCoTGenerator:
    def __init__(
        self,
        num_entities: int = 1000,
        num_concepts: int = 200,
        avg_connections_per_entity: int = 8,
        graph_connectivity: float = 0.15,
        reasoning_depths: List[int] = [3, 4, 5, 6],
        seed: int = 42
    ):
        """
        Initialize the shared graph CoT generator.
        
        Args:
            num_entities: Number of entity nodes in the graph
            num_concepts: Number of concept nodes in the graph  
            avg_connections_per_entity: Average number of edges per entity
            graph_connectivity: Probability of random connections between nodes
            reasoning_depths: List of possible reasoning depths to sample from
            seed: Random seed for reproducibility
        """
        self.num_entities = num_entities
        self.num_concepts = num_concepts
        self.avg_connections_per_entity = avg_connections_per_entity
        self.graph_connectivity = graph_connectivity
        self.reasoning_depths = reasoning_depths
        self.seed = seed
        
        # Initialize random generators
        random.seed(seed)
        np.random.seed(seed)
        
        # Graph structure: adjacency list representation
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)
        self.all_edges: List[Tuple[str, str]] = []
        
        # Node sets
        self.entities: List[str] = []
        self.concepts: List[str] = []
        self.all_nodes: List[str] = []
        
        # Build the shared graph
        self._build_graph()
        
    def _build_graph(self):
        """Build the large shared graph with entities and concepts."""
        print("Building shared graph...")
        
        # Create entity nodes
        self.entities = [f"entity_{i:04d}" for i in range(self.num_entities)]
        
        # Create concept nodes  
        self.concepts = [f"concept_{i:03d}" for i in range(self.num_concepts)]
        
        # All nodes
        self.all_nodes = self.entities + self.concepts
        
        # Build hierarchical structure: entities connect to concepts, concepts to concepts
        self._add_entity_concept_connections()
        self._add_concept_concept_connections()
        self._add_random_connections()
        
        print(f"Graph built with {len(self.all_nodes)} nodes and {len(self.all_edges)} edges")
        
    def _add_entity_concept_connections(self):
        """Add connections from entities to concepts (bottom-up)."""
        for entity in self.entities:
            # Each entity connects to 2-5 concepts
            num_connections = random.randint(2, 5)
            connected_concepts = random.sample(self.concepts, num_connections)
            
            for concept in connected_concepts:
                self._add_edge(entity, concept)
                
    def _add_concept_concept_connections(self):
        """Add hierarchical connections between concepts."""
        # Create concept hierarchy - some concepts are more general than others
        concept_levels = {}
        
        # Assign concepts to levels (0=most specific, higher=more general)
        for i, concept in enumerate(self.concepts):
            concept_levels[concept] = i % 4  # 4 levels of hierarchy
            
        # Connect concepts: lower level -> higher level with some probability
        for concept1 in self.concepts:
            level1 = concept_levels[concept1]
            
            # Connect to 1-3 concepts at higher levels
            higher_level_concepts = [c for c in self.concepts 
                                   if concept_levels[c] > level1 and c != concept1]
            
            if higher_level_concepts:
                num_connections = random.randint(1, min(3, len(higher_level_concepts)))
                connected = random.sample(higher_level_concepts, num_connections)
                
                for concept2 in connected:
                    if random.random() < 0.7:  # 70% chance of connection
                        self._add_edge(concept1, concept2)
                        
    def _add_random_connections(self):
        """Add some random connections to increase graph connectivity."""
        total_possible = len(self.all_nodes) * (len(self.all_nodes) - 1)
        num_random = int(total_possible * self.graph_connectivity * 0.1)  # 10% of connectivity budget
        
        for _ in range(num_random):
            node1, node2 = random.sample(self.all_nodes, 2)
            if node2 not in self.graph[node1] and random.random() < 0.3:
                self._add_edge(node1, node2)
                
    def _add_edge(self, from_node: str, to_node: str):
        """Add a directed edge to the graph."""
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)
            self.reverse_graph[to_node].append(from_node)
            self.all_edges.append((from_node, to_node))
            
    def _find_path(self, start: str, end: str, max_depth: int) -> Optional[List[str]]:
        """Find a path from start to end node within max_depth using BFS."""
        if start == end:
            return [start]
            
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
                
            for neighbor in self.graph[current]:
                if neighbor == end:
                    return path + [neighbor]
                    
                if neighbor not in visited and len(path) < max_depth:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        return None
        
    def _sample_reasoning_path(self, depth: int) -> Optional[Tuple[List[str], str, str]]:
        """
        Sample a reasoning path of specified depth.
        
        Returns:
            Tuple of (path, question_type, answer) or None if no valid path found
        """
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Start with an entity
            start_entity = random.choice(self.entities)
            
            # Try to find a path of the specified depth
            # We'll look for paths that end in concepts
            target_concepts = random.sample(self.concepts, min(10, len(self.concepts)))
            
            for target_concept in target_concepts:
                path = self._find_path(start_entity, target_concept, depth)
                
                if path and len(path) == depth + 1:  # depth+1 because path includes start and end
                    # Create binary choice question
                    # Choose another concept as foil
                    foil_concepts = [c for c in self.concepts if c != target_concept]
                    foil_concept = random.choice(foil_concepts)
                    
                    # Randomly decide correct answer position
                    if random.random() < 0.5:
                        question_type = f"Is {start_entity} associated with {target_concept} or {foil_concept}?"
                        answer = target_concept
                    else:
                        question_type = f"Is {start_entity} associated with {foil_concept} or {target_concept}?"
                        answer = target_concept
                        
                    return path, question_type, answer
                    
        return None
        
    def _generate_reasoning_steps(self, path: List[str]) -> List[str]:
        """Generate human-readable reasoning steps from a path."""
        steps = []
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Create reasoning step
            if from_node.startswith("entity_"):
                from_desc = f"Entity {from_node.split('_')[1]}"
            else:
                from_desc = f"Concept {from_node.split('_')[1]}"
                
            if to_node.startswith("entity_"):
                to_desc = f"Entity {to_node.split('_')[1]}"
            else:
                to_desc = f"Concept {to_node.split('_')[1]}"
                
            step = f"{from_desc} is connected to {to_desc}."
            steps.append(step)
            
        return steps
        
    def _sample_context_edges(
        self, 
        reasoning_path: List[str], 
        num_context_edges: int,
        bias_factor: float = 3.0
    ) -> List[Tuple[str, str]]:
        """
        Sample context edges with bias towards edges close to reasoning path.
        
        Args:
            reasoning_path: The sampled reasoning path nodes
            num_context_edges: Number of context edges to sample
            bias_factor: How much to bias towards path-adjacent edges
            
        Returns:
            List of context edges
        """
        path_nodes = set(reasoning_path)
        
        # Categorize edges by their relationship to the reasoning path
        path_adjacent_edges = []  # Edges with one node in the path
        path_internal_edges = []  # Edges with both nodes in the path (rare)
        other_edges = []          # Edges with no nodes in the path
        
        for edge in self.all_edges:
            from_node, to_node = edge
            from_in_path = from_node in path_nodes
            to_in_path = to_node in path_nodes
            
            if from_in_path and to_in_path:
                path_internal_edges.append(edge)
            elif from_in_path or to_in_path:
                path_adjacent_edges.append(edge)
            else:
                other_edges.append(edge)
                
        # Sample with bias
        context_edges = []
        
        # Always include some path-adjacent edges if available
        if path_adjacent_edges:
            num_adjacent = min(num_context_edges // 2, len(path_adjacent_edges))
            context_edges.extend(random.sample(path_adjacent_edges, num_adjacent))
            
        # Include some path-internal edges if available
        if path_internal_edges:
            num_internal = min((num_context_edges - len(context_edges)) // 3, len(path_internal_edges))
            context_edges.extend(random.sample(path_internal_edges, num_internal))
            
        # Fill remaining with random edges
        remaining_needed = num_context_edges - len(context_edges)
        if remaining_needed > 0 and other_edges:
            num_random = min(remaining_needed, len(other_edges))
            context_edges.extend(random.sample(other_edges, num_random))
            
        # If we still need more, sample from all available edges
        if len(context_edges) < num_context_edges:
            all_available = [e for e in self.all_edges if e not in context_edges]
            additional_needed = num_context_edges - len(context_edges)
            if all_available:
                num_additional = min(additional_needed, len(all_available))
                context_edges.extend(random.sample(all_available, num_additional))
                
        return context_edges[:num_context_edges]
        
    def generate_sample(self, num_context_edges: int = 20) -> Optional[Dict[str, Any]]:
        """
        Generate a single CoT reasoning sample.
        
        Args:
            num_context_edges: Number of context edges to include
            
        Returns:
            Dictionary with question, steps, answer, and metadata
        """
        # Sample reasoning depth
        depth = random.choice(self.reasoning_depths)
        
        # Sample reasoning path
        path_result = self._sample_reasoning_path(depth)
        if path_result is None:
            return None
            
        path, question, answer = path_result
        
        # Generate reasoning steps
        steps = self._generate_reasoning_steps(path)
        
        # Sample context edges (not used in output but influences generation)
        context_edges = self._sample_context_edges(path, num_context_edges)
        
        return {
            "question": question,
            "steps": steps,
            "answer": answer,
            "metadata": {
                "reasoning_path": path,
                "reasoning_depth": depth,
                "num_context_edges": len(context_edges),
                "context_edges": context_edges  # For debugging/analysis
            }
        }
        
    def generate_dataset(
        self,
        num_samples: int,
        num_context_edges: int = 20,
        output_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Number of samples to generate
            num_context_edges: Number of context edges per sample
            output_path: Optional path to save JSON output
            
        Returns:
            List of generated samples
        """
        print(f"Generating {num_samples} samples...")
        
        samples = []
        failed_attempts = 0
        max_failed_attempts = num_samples * 10  # Allow some failures
        
        with_depth_counts = {depth: 0 for depth in self.reasoning_depths}
        
        while len(samples) < num_samples and failed_attempts < max_failed_attempts:
            sample = self.generate_sample(num_context_edges)
            
            if sample is not None:
                samples.append(sample)
                depth = sample["metadata"]["reasoning_depth"]
                with_depth_counts[depth] += 1
                
                if len(samples) % 1000 == 0:
                    print(f"Generated {len(samples)}/{num_samples} samples...")
            else:
                failed_attempts += 1
                
        print(f"Generated {len(samples)} samples with {failed_attempts} failed attempts")
        print(f"Depth distribution: {with_depth_counts}")
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"Saved dataset to {output_path}")
            
        return samples
        
    def export_to_txt_format(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        num_train: int = 8000,
        num_val: int = 1000
    ):
        """
        Export samples to the ProsQA/ProntoQA txt format.
        
        Format: question||steps + " #### " + answer
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert samples to txt format
        txt_lines = []
        for sample in samples:
            question = sample["question"]
            steps = sample["steps"]
            answer = sample["answer"]
            
            # Join steps and add answer
            cot = ' '.join(steps) + f" #### {answer}"
            txt_lines.append(f"{question}||{cot}\n")
            
        # Split into train/val/test
        bounds = [(0, num_train), (num_train, num_train + num_val), (num_train + num_val, len(txt_lines))]
        
        for phase, (l, r) in zip(["train", "valid", "test"], bounds):
            out_fname = f"shared_graph_cot_{phase}.txt"
            out_path = os.path.join(output_dir, out_fname)
            
            with open(out_path, "w") as out:
                for i in range(l, min(r, len(txt_lines))):
                    out.write(txt_lines[i])
                    
            actual_samples = min(r, len(txt_lines)) - l
            print(f"Wrote {actual_samples} {phase} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate CoT dataset with shared implicit graph")
    
    # Graph structure parameters
    parser.add_argument("--num_entities", type=int, default=1000, 
                       help="Number of entity nodes in the graph")
    parser.add_argument("--num_concepts", type=int, default=200,
                       help="Number of concept nodes in the graph") 
    parser.add_argument("--graph_connectivity", type=float, default=0.15,
                       help="Graph connectivity factor")
    
    # Dataset generation parameters
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--num_context_edges", type=int, default=20,
                       help="Number of context edges per sample")
    parser.add_argument("--reasoning_depths", type=int, nargs="+", default=[3, 4, 5, 6],
                       help="Possible reasoning depths")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for generated files")
    parser.add_argument("--num_train", type=int, default=8000,
                       help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=1000,
                       help="Number of validation samples")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_json", action="store_true",
                       help="Also save JSON format with metadata")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SharedGraphCoTGenerator(
        num_entities=args.num_entities,
        num_concepts=args.num_concepts,
        graph_connectivity=args.graph_connectivity,
        reasoning_depths=args.reasoning_depths,
        seed=args.seed
    )
    
    # Generate dataset
    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        num_context_edges=args.num_context_edges
    )
    
    # Export to txt format (compatible with existing pipeline)
    generator.export_to_txt_format(
        samples=samples,
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val
    )
    
    # Optionally save JSON with metadata
    if args.save_json:
        json_path = os.path.join(args.output_dir, "shared_graph_cot_full.json")
        with open(json_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Saved full dataset with metadata to {json_path}")


if __name__ == "__main__":
    main()
