import pickle
import networkx as nx
import os
import numpy as np
import torch
import argparse
import sys
import matplotlib.pyplot as plt

def draw_molecule(G):
    """
    Visualize a molecular graph with atom types and bond types.
    
    Args:
        G: NetworkX graph with 'symbol' node attribute and 'bond_type' edge attribute
    """
    from matplotlib.lines import Line2D
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Color nodes by atom type
    atom_colors = {
        'C': '#404040',  # Carbon - dark gray
        'N': '#3050F8',  # Nitrogen - blue
        'O': '#FF0D0D',  # Oxygen - red
        'S': '#FFFF30',  # Sulfur - yellow
        'P': '#FF8000',  # Phosphorus - orange
        'F': '#90E050',  # Fluorine - light green
        'Cl': '#1FF01F', # Chlorine - green
        'Br': '#A62929', # Bromine - brown
        'I': '#940094',  # Iodine - purple
    }
    
    node_colors = []
    node_labels = {}
    for node in G.nodes():
        symbol = G.nodes[node].get('symbol', 'C')
        node_colors.append(atom_colors.get(symbol, '#808080'))
        node_labels[node] = symbol
    
    # Draw edges by bond type
    bond_styles = {
        'SINGLE': {'width': 2, 'style': 'solid', 'color': '#000000'},
        'DOUBLE': {'width': 4, 'style': 'solid', 'color': '#000000'},
        'TRIPLE': {'width': 6, 'style': 'solid', 'color': '#000000'},
        'AROMATIC': {'width': 2, 'style': 'dashed', 'color': '#FF00FF'}
    }
    
    # Group edges by type
    edges_by_type = {}
    for u, v, d in G.edges(data=True):
        bond_type = d.get('bond_type', 'SINGLE')
        if bond_type not in edges_by_type:
            edges_by_type[bond_type] = []
        edges_by_type[bond_type].append((u, v))
    
    # Draw each bond type separately
    for bond_type, edges in edges_by_type.items():
        style = bond_styles.get(bond_type, bond_styles['SINGLE'])
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            width=style['width'],
            style=style['style'],
            edge_color=style['color'],
            alpha=0.7
        )
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=12, font_weight='bold', font_color='white')
    
    # Create legend
    legend_elements = []
    
    # Atom type legend
    unique_atoms = set(G.nodes[n].get('symbol', 'C') for n in G.nodes())
    for atom in sorted(unique_atoms):
        if atom in atom_colors:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=atom_colors[atom], markersize=10, label=f'{atom} atom')
            )
    
    # Bond type legend
    for bond_type in sorted(edges_by_type.keys()):
        style = bond_styles.get(bond_type, bond_styles['SINGLE'])
        legend_elements.append(
            Line2D([0], [0], color=style['color'], linewidth=style['width']/2,
                   linestyle=style['style'], label=f'{bond_type} bond')
        )
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Molecular Structure', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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
        
        print("\nNode Attribute Details:")
        for key, val in node_attrs.items():
            if isinstance(val, (list, np.ndarray, torch.Tensor)):
                try:
                    shape_info = val.shape if hasattr(val, 'shape') else len(val)
                    print(f"  - '{key}': Type={type(val).__name__}, Shape/Len={shape_info}")
                    
                    flat_val = np.array(val).flatten()
                    if len(flat_val) > 0 and isinstance(flat_val[0], (int, float, np.number)):
                        sample = flat_val[:5]
                        print(f"    Sample: {sample}...")
                except:
                    print(f"  - '{key}': Type={type(val).__name__} (Could not determine shape)")
            else:
                 print(f"  - '{key}': Type={type(val).__name__}, Value={val}")
        
        # Inspect edges
        if G.number_of_edges() > 0:
            first_edge = list(G.edges())[0]
            edge_attrs = G.edges[first_edge]
            print(f"\nSample Edge: {first_edge}")
            print(f"Edge Attributes Keys: {list(edge_attrs.keys())}")
            
            print("\nEdge Attribute Details:")
            for key, val in edge_attrs.items():
                if isinstance(val, (list, np.ndarray, torch.Tensor)):
                    try:
                        shape_info = val.shape if hasattr(val, 'shape') else len(val)
                        print(f"  - '{key}': Type={type(val).__name__}, Shape/Len={shape_info}")
                    except:
                        print(f"  - '{key}': Type={type(val).__name__}")
                else:
                    print(f"  - '{key}': Type={type(val).__name__}, Value={val}")
            
            # Check for edge type diversity
            edge_types = set()
            for u, v, d in G.edges(data=True):
                if 'order' in d:
                    edge_types.add(d['order'])
            if edge_types:
                print(f"\nUnique edge types (bond orders): {sorted(edge_types)}")

            # Visualize as molecular structure
            draw_molecule(G)

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
