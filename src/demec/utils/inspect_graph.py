import pickle
import networkx as nx
import os
import numpy as np
import torch
import argparse
import sys

def inspect_graph(graph_path):
    if not os.path.exists(graph_path):
        print(f"Error: File not found at {graph_path}")
        return

    print(f"Inspecting graph: {graph_path}")
    print("-" * 50)
    
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        print(f"Type: {type(G)}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        
        if G.number_of_nodes() == 0:
            print("Graph is empty.")
            return

        # Get the first node index (it might not be 0)
        first_node = list(G.nodes())[0]
        print(f"Sample Node ID: {first_node}")
        
        node_attrs = G.nodes[first_node]
        print(f"Node Attributes Keys: {list(node_attrs.keys())}")
        
        print("\nAttribute Details:")
        for key, val in node_attrs.items():
            if isinstance(val, (list, np.ndarray, torch.Tensor)):
                try:
                    # Handle both list/array and tensor len/shape
                    shape_info = val.shape if hasattr(val, 'shape') else len(val)
                    print(f"  - '{key}': Type={type(val).__name__}, Shape/Len={shape_info}")
                    
                    # Print a small sample if it's numerical features
                    flat_val = np.array(val).flatten()
                    if len(flat_val) > 0 and isinstance(flat_val[0], (int, float, np.number)):
                        sample = flat_val[:5]
                        print(f"    Sample: {sample}...")
                except:
                    print(f"  - '{key}': Type={type(val).__name__} (Could not determine shape)")
            else:
                 print(f"  - '{key}': Type={type(val).__name__}, Value={val}")

    except Exception as e:
        print(f"Error reading graph: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect attributes of a pickled NetworkX graph.")
    parser.add_argument("path", help="Path to the .gpickle file to inspect")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    inspect_graph(args.path)

if __name__ == "__main__":
    main()
