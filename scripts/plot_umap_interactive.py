#!/usr/bin/env python3
"""
Interactive UMAP visualization with Plotly.

Creates an interactive HTML plot where you can:
- Hover over points to see molecule details
- Zoom and pan
- Click to see full side effect list
- Filter by side effect count

Usage:
    python scripts/plot_umap_interactive.py --checkpoint checkpoints/model_best.pt
"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import umap
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader

from demec.data.data_loader import GraphStructureDataset
from demec.models.gnn_backbone import GNNBackbone


def extract_embeddings(checkpoint_path, graphs_dir='data/processed/graphs_v2/', batch_size=32):
    """Extract embeddings from a trained model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    
    # Handle both dict and object formats for args
    def get_arg(args, key, default=None):
        if isinstance(args, dict):
            return args.get(key, default)
        else:
            return getattr(args, key, default)
    
    # Load dataset
    print(f"Loading dataset from: {graphs_dir}")
    task_config = {
        'side_effects': "data/processed/cid_se_matrix.csv",
        'atc': "data/processed/cid_atc_l3_matrix.csv",
        'maccs': "data/processed/cid_maccs_matrix.csv",
        'molprops': "data/processed/cid_molprops_matrix.csv"
    }
    
    dataset = GraphStructureDataset(
        graph_dir=graphs_dir,
        task_config=task_config,
        feature_key=get_arg(args, 'feature_key', None)
    )
    
    # Create backbone model
    print("Creating model...")
    backbone = GNNBackbone(
        input_dim=get_arg(args, 'input_dim', 1),
        hidden_dim=get_arg(args, 'hidden_dim', 64),
        num_layers=get_arg(args, 'num_layers', 5),
        dropout=get_arg(args, 'dropout', 0.2),
        conv_type=get_arg(args, 'model', 'gcn'),
        heads=get_arg(args, 'heads', 3)
    )
    
    # Load weights
    full_model_state = checkpoint['model_state_dict']
    backbone_state = {k.replace('backbone.', ''): v 
                     for k, v in full_model_state.items() 
                     if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    
    # Extract embeddings
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    all_cids = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            embeddings = backbone(batch)
            all_embeddings.append(embeddings.cpu().numpy())
            all_cids.extend(batch.cid.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    cids = np.array(all_cids)
    
    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    return embeddings, cids


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """Compute UMAP projection."""
    print(f"\nComputing UMAP projection...")
    print(f"  n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    
    umap_embeddings = reducer.fit_transform(embeddings)
    print("UMAP complete!")
    
    return umap_embeddings


def load_molecule_info(cids):
    """Load detailed information about molecules."""
    print("\nLoading molecule information...")
    
    # Load side effects matrix
    se_df = pd.read_csv('data/processed/cid_se_matrix.csv').set_index('cid')
    
    # Load side effect ID to name mapping
    se_name_dict = {}
    if os.path.exists('data/processed/edges.csv'):
        print("Loading side effect names...")
        edges_df = pd.read_csv('data/processed/edges.csv')
        # Create mapping from se_id to se_name
        se_name_dict = dict(zip(edges_df['se_id'], edges_df['se_name']))
        print(f"Loaded names for {len(se_name_dict)} side effects")
    else:
        print("Side effect names file not found")
    
    # Load SMILES and drug names from cache
    smiles_dict = {}
    name_dict = {}
    if os.path.exists('data/processed/smiles_cache.csv'):
        print("Loading drug names and SMILES from cache...")
        smiles_df = pd.read_csv('data/processed/smiles_cache.csv')
        smiles_dict = dict(zip(smiles_df['cid'], smiles_df['smiles_sanitized']))
        name_dict = dict(zip(smiles_df['cid'], smiles_df['name']))
        print(f"Loaded data for {len(smiles_dict)} compounds")
    else:
        print("SMILES cache not found")
    
    # Build molecule info
    molecule_info = []
    for cid in tqdm(cids, desc="Processing molecules"):
        info = {'cid': int(cid)}
        
        # Get drug name
        info['name'] = name_dict.get(int(cid), f'CID {int(cid)}')
        
        # Create PubChem link
        info['pubchem_url'] = f'https://pubchem.ncbi.nlm.nih.gov/compound/{int(cid)}'
        
        # Get SMILES
        info['smiles'] = smiles_dict.get(int(cid), 'N/A')
        
        # Get side effects
        if cid in se_df.index:
            se_vector = se_df.loc[cid]
            active_se_ids = se_vector[se_vector > 0].index.tolist()
            info['n_side_effects'] = len(active_se_ids)
            
            # Convert SE IDs to human-readable names
            active_se_names = [se_name_dict.get(se_id, se_id) for se_id in active_se_ids]
            info['side_effects'] = active_se_names
            
            # Create a short preview (top 5) with names
            info['se_preview'] = ', '.join(active_se_names[:5])
            if len(active_se_names) > 5:
                info['se_preview'] += f' (+{len(active_se_names)-5} more)'
        else:
            info['n_side_effects'] = 0
            info['side_effects'] = []
            info['se_preview'] = 'None'
        
        molecule_info.append(info)
    
    return pd.DataFrame(molecule_info)


def create_interactive_plot(umap_emb, mol_df, output_path):
    """Create interactive Plotly visualization."""
    print("\nCreating interactive plot...")
    
    # Create bins for coloring
    n_se = mol_df['n_side_effects'].values
    bins = [0, 5, 10, 20, 50, 100, n_se.max()+1]
    labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '100+']
    mol_df['se_category'] = pd.cut(n_se, bins=bins, labels=labels)
    
    mol_df['umap_1'] = umap_emb[:, 0]
    mol_df['umap_2'] = umap_emb[:, 1]
    
    # Create figure with Plotly Express
    fig = px.scatter(
        mol_df,
        x='umap_1',
        y='umap_2',
        color='se_category',
        color_discrete_sequence=px.colors.sequential.Viridis,
        hover_data={'umap_1': False, 'umap_2': False, 'se_category': False},
        custom_data=['name', 'cid', 'pubchem_url', 'n_side_effects', 'smiles', 'se_preview'],
        title='Interactive UMAP: Learned Molecular Embeddings',
        labels={'umap_1': 'UMAP Dimension 1', 'umap_2': 'UMAP Dimension 2'},
        category_orders={'se_category': labels}
    )
    
    # Update hover template with clickable link
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      '<b>CID:</b> <a href="%{customdata[2]}" target="_blank">%{customdata[1]}</a><br>' +
                      '<b>Side Effects:</b> %{customdata[3]}<br>' +
                      '<b>SMILES:</b> %{customdata[4]:.50}<br>' +
                      '<b>Top SEs:</b> %{customdata[5]}<br>' +
                      '<i>Click CID to open PubChem</i>' +
                      '<extra></extra>',
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white'))
    )
    
    # Update layout
    fig.update_layout(
        width=1400,
        height=900,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            title=dict(text='# Side Effects', font=dict(size=14)),
            font=dict(size=12),
            x=1.02,
            y=0.5
        ),
        hovermode='closest',
        title=dict(
            text='Interactive UMAP: Learned Molecular Embeddings<br><sub>Hover over points for details | Click and drag to zoom | Double-click to reset</sub>',
            font=dict(size=18)
        )
    )
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Saved interactive plot: {output_path}")
    
    # Also create a continuous version
    output_continuous = output_path.replace('.html', '_continuous.html')
    
    fig2 = px.scatter(
        mol_df,
        x='umap_1',
        y='umap_2',
        color='n_side_effects',
        color_continuous_scale='Viridis',
        hover_data={'umap_1': False, 'umap_2': False, 'n_side_effects': False},
        custom_data=['name', 'cid', 'pubchem_url', 'n_side_effects', 'smiles', 'se_preview'],
        title='Interactive UMAP: Continuous Coloring by Side Effect Count',
        labels={'umap_1': 'UMAP Dimension 1', 'umap_2': 'UMAP Dimension 2', 'n_side_effects': '# Side Effects'}
    )
    
    fig2.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      '<b>CID:</b> <a href="%{customdata[2]}" target="_blank">%{customdata[1]}</a><br>' +
                      '<b>Side Effects:</b> %{customdata[3]}<br>' +
                      '<b>SMILES:</b> %{customdata[4]:.50}<br>' +
                      '<b>Top SEs:</b> %{customdata[5]}<br>' +
                      '<i>Click CID to open PubChem</i>' +
                      '<extra></extra>',
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white'))
    )
    
    fig2.update_layout(
        width=1400,
        height=900,
        template='plotly_white',
        font=dict(size=12),
        hovermode='closest',
        title=dict(
            text='Interactive UMAP: Continuous Coloring by Side Effect Count<br><sub>Hover over points for details | Click and drag to zoom | Double-click to reset</sub>',
            font=dict(size=18)
        )
    )
    
    fig2.write_html(output_continuous)
    print(f"Saved continuous plot: {output_continuous}")
    
    return fig, fig2


def create_detailed_table(mol_df, output_path):
    """Create a detailed table with all molecule information."""
    print("\nCreating detailed data table...")
    
    # Create a CSV with full side effect lists
    export_df = mol_df[['cid', 'name', 'pubchem_url', 'n_side_effects', 'smiles']].copy()
    export_df['side_effects_list'] = mol_df['side_effects'].apply(lambda x: '; '.join(x) if x else 'None')
    
    csv_path = output_path.replace('.html', '_data.csv')
    export_df.to_csv(csv_path, index=False)
    print(f"Saved detailed data: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Create interactive UMAP visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--graphs_dir', type=str, default='data/processed/graphs_v2/',
                       help='Directory containing molecular graphs')
    parser.add_argument('--output_dir', type=str, default='results/umap_interactive',
                       help='Output directory for visualizations')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='UMAP n_neighbors (5-50)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist (0.0-0.99)')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['euclidean', 'cosine', 'manhattan', 'correlation'],
                       help='Distance metric for UMAP')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract embeddings
    embeddings, cids = extract_embeddings(
        args.checkpoint,
        args.graphs_dir,
        args.batch_size
    )
    
    # Compute UMAP
    umap_embeddings = compute_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )
    
    # Load molecule information
    mol_df = load_molecule_info(cids)
    
    # Create interactive plots
    output_path = os.path.join(args.output_dir, 'umap_interactive.html')
    fig, fig2 = create_interactive_plot(umap_embeddings, mol_df, output_path)
    
    # Create detailed table
    create_detailed_table(mol_df, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Interactive visualization complete!")
    print(f"{'='*60}")
    print(f"\n📊 Results:")
    print(f"  - Interactive plot (categorical): {output_path}")
    print(f"  - Interactive plot (continuous): {output_path.replace('.html', '_continuous.html')}")
    print(f"  - Detailed data CSV: {output_path.replace('.html', '_data.csv')}")
    print(f"\n💡 To view:")
    print(f"  open {output_path}")
    print(f"\n✨ Features:")
    print(f"  - Hover over points to see molecule details")
    print(f"  - Click and drag to zoom")
    print(f"  - Double-click to reset view")
    print(f"  - Click legend items to hide/show categories")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
