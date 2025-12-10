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

from demec.data.data_loader import GraphDataset
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
    
    feature_key = get_arg(args, 'feature_key', 'atom_features')
    
    # Load dataset
    print(f"Loading dataset from: {graphs_dir}")
    task_config = {
        'side_effects': "data/processed/cid_se_matrix.csv",
        'atc': "data/processed/cid_atc_l3_matrix.csv",
        'maccs': "data/processed/cid_maccs_matrix.csv",
        'molprops': "data/processed/cid_molprops_matrix.csv"
    }
    
    dataset = GraphDataset(
        graph_dir=graphs_dir,
        task_config=task_config,
        feature_key=feature_key
    )
    
    # Create backbone model
    print("Creating model...")
    backbone = GNNBackbone(
        input_dim=get_arg(args, 'input_dim', 154),
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
    
    return embeddings, cids


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """Compute UMAP projection."""
    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    
    umap_embeddings = reducer.fit_transform(embeddings)
    
    return umap_embeddings


def load_molecule_info(cids, max_side_effects=None):
    """Load detailed information about molecules."""
    print("Loading molecule information...")
    
    # Load side effects matrix
    se_df = pd.read_csv('data/processed/cid_se_matrix.csv').set_index('cid')
    
    # Filter to top N side effects if specified (to match model training)
    if max_side_effects is not None:
        col_sums = se_df.sum(axis=0).sort_values(ascending=False)
        top_cols = col_sums.head(max_side_effects).index.tolist()
        se_df = se_df[top_cols]
    
    # Load side effect ID to name mapping
    se_name_dict = {}
    if os.path.exists('data/processed/edges.csv'):
        edges_df = pd.read_csv('data/processed/edges.csv')
        se_name_dict = dict(zip(edges_df['se_id'], edges_df['se_name']))
    
    # Load SMILES and drug names from cache
    smiles_dict = {}
    name_dict = {}
    if os.path.exists('data/processed/smiles_cache.csv'):
        smiles_df = pd.read_csv('data/processed/smiles_cache.csv')
        smiles_dict = dict(zip(smiles_df['cid'], smiles_df['smiles_sanitized']))
        name_dict = dict(zip(smiles_df['cid'], smiles_df['name']))
    
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
    """Create interactive Plotly visualization with slider."""
    print("Creating interactive plot...")
    
    mol_df['umap_1'] = umap_emb[:, 0]
    mol_df['umap_2'] = umap_emb[:, 1]
    
    # Create version with range slider for filtering
    output_slider = output_path
    
    import plotly.graph_objects as go
    
    # Create figure with slider
    fig3 = go.Figure()
    
    # Add all points initially
    fig3.add_trace(go.Scatter(
        x=mol_df['umap_1'],
        y=mol_df['umap_2'],
        mode='markers',
        marker=dict(
            size=8,
            color=mol_df['n_side_effects'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='# Side Effects'),
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        customdata=mol_df[['name', 'cid', 'pubchem_url', 'n_side_effects', 'smiles', 'se_preview']].values,
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      '<b>CID:</b> <a href="%{customdata[2]}" target="_blank">%{customdata[1]}</a><br>' +
                      '<b>Side Effects:</b> %{customdata[3]}<br>' +
                      '<b>SMILES:</b> %{customdata[4]:.50}<br>' +
                      '<b>Top SEs:</b> %{customdata[5]}<br>' +
                      '<i>Click CID to open PubChem</i>' +
                      '<extra></extra>',
        name='All Drugs'
    ))
    
    # Create steps for the slider
    min_se = int(mol_df['n_side_effects'].min())
    max_se = int(mol_df['n_side_effects'].max())
    
    steps = []
    for se_min in range(min_se, max_se + 1, 1):
        for se_max in range(se_min, max_se + 1, 1):
            # Filter data
            mask = (mol_df['n_side_effects'] >= se_min) & (mol_df['n_side_effects'] <= se_max)
            
            step = dict(
                method="update",
                args=[{"visible": [True]},
                      {"title": f"UMAP: Drugs with {se_min}-{se_max} Side Effects ({mask.sum()} drugs)"}],
                label=f"{se_min}-{se_max}"
            )
            steps.append(step)
    
    # Add range slider
    fig3.update_layout(
        width=1400,
        height=1000,
        template='plotly_white',
        font=dict(size=12),
        hovermode='closest',
        title=dict(
            text=f'Interactive UMAP with Range Filter<br><sub>Use sliders below to filter by side effect count | Showing all {len(mol_df)} drugs</sub>',
            font=dict(size=18)
        ),
        xaxis=dict(title='UMAP Dimension 1'),
        yaxis=dict(title='UMAP Dimension 2'),
        sliders=[{
            'active': max_se - min_se,
            'yanchor': 'top',
            'y': -0.1,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Side Effect Range: ',
                'visible': True,
                'xanchor': 'right'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'steps': steps
        }]
    )
    
    # Add custom JavaScript for dual range slider
    fig3_html = fig3.to_html(include_plotlyjs='cdn')
    
    # Create custom HTML with dual range slider
    custom_html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .controls {{
            margin: 20px 0;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .slider-container {{
            margin: 15px 0;
        }}
        .slider-label {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        input[type="range"] {{
            width: 45%;
            margin: 0 10px;
        }}
        .value-display {{
            font-size: 18px;
            color: #333;
            margin: 10px 0;
        }}
        #plot {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Interactive UMAP: Filter by Side Effect Count</h1>
    
    <div class="controls">
        <div class="slider-container">
            <div class="slider-label">Minimum Side Effects:</div>
            <input type="range" id="minSlider" min="{min_se}" max="{max_se}" value="{min_se}" step="1">
            <span id="minValue">{min_se}</span>
        </div>
        
        <div class="slider-container">
            <div class="slider-label">Maximum Side Effects:</div>
            <input type="range" id="maxSlider" min="{min_se}" max="{max_se}" value="{max_se}" step="1">
            <span id="maxValue">{max_se}</span>
        </div>
        
        <div class="value-display">
            Showing drugs with <span id="rangeDisplay">{min_se} to {max_se}</span> side effects
            (<span id="countDisplay">{len(mol_df)}</span> drugs)
        </div>
    </div>
    
    <div id="plot"></div>
    
    <script>
        // Data
        const allData = {{
            x: {mol_df['umap_1'].tolist()},
            y: {mol_df['umap_2'].tolist()},
            se_counts: {mol_df['n_side_effects'].tolist()},
            names: {mol_df['name'].tolist()},
            cids: {mol_df['cid'].tolist()},
            urls: {mol_df['pubchem_url'].tolist()},
            smiles: {mol_df['smiles'].tolist()},
            se_preview: {mol_df['se_preview'].tolist()}
        }};
        
        const minSlider = document.getElementById('minSlider');
        const maxSlider = document.getElementById('maxSlider');
        const minValue = document.getElementById('minValue');
        const maxValue = document.getElementById('maxValue');
        const rangeDisplay = document.getElementById('rangeDisplay');
        const countDisplay = document.getElementById('countDisplay');
        
        function updatePlot() {{
            const minSE = parseInt(minSlider.value);
            const maxSE = parseInt(maxSlider.value);
            
            // Ensure min <= max
            if (minSE > maxSE) {{
                if (this.id === 'minSlider') {{
                    maxSlider.value = minSE;
                }} else {{
                    minSlider.value = maxSE;
                }}
                return updatePlot();
            }}
            
            // Update displays
            minValue.textContent = minSE;
            maxValue.textContent = maxSE;
            rangeDisplay.textContent = minSE + ' to ' + maxSE;
            
            // Filter data
            const filteredIndices = [];
            for (let i = 0; i < allData.se_counts.length; i++) {{
                if (allData.se_counts[i] >= minSE && allData.se_counts[i] <= maxSE) {{
                    filteredIndices.push(i);
                }}
            }}
            
            countDisplay.textContent = filteredIndices.length;
            
            // Create filtered arrays
            const filteredX = filteredIndices.map(i => allData.x[i]);
            const filteredY = filteredIndices.map(i => allData.y[i]);
            const filteredColors = filteredIndices.map(i => allData.se_counts[i]);
            const filteredCustomData = filteredIndices.map(i => [
                allData.names[i],
                allData.cids[i],
                allData.urls[i],
                allData.se_counts[i],
                allData.smiles[i],
                allData.se_preview[i]
            ]);
            
            // Create trace
            const trace = {{
                x: filteredX,
                y: filteredY,
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 8,
                    color: filteredColors,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{title: '# Side Effects'}},
                    opacity: 0.7,
                    line: {{width: 0.5, color: 'white'}}
                }},
                customdata: filteredCustomData,
                hovertemplate: '<b>%{{customdata[0]}}</b><br>' +
                              '<b>CID:</b> <a href="%{{customdata[2]}}" target="_blank">%{{customdata[1]}}</a><br>' +
                              '<b>Side Effects:</b> %{{customdata[3]}}<br>' +
                              '<b>SMILES:</b> %{{customdata[4]}}<br>' +
                              '<b>Top SEs:</b> %{{customdata[5]}}<br>' +
                              '<i>Click CID to open PubChem</i>' +
                              '<extra></extra>'
            }};
            
            const layout = {{
                width: 1400,
                height: 900,
                template: 'plotly_white',
                hovermode: 'closest',
                xaxis: {{title: 'UMAP Dimension 1'}},
                yaxis: {{title: 'UMAP Dimension 2'}},
                title: {{
                    text: 'Interactive UMAP: Learned Molecular Embeddings<br><sub>Adjust sliders to filter by side effect count</sub>',
                    font: {{size: 18}}
                }}
            }};
            
            Plotly.newPlot('plot', [trace], layout);
        }}
        
        minSlider.addEventListener('input', updatePlot);
        maxSlider.addEventListener('input', updatePlot);
        
        // Initial plot
        updatePlot();
    </script>
</body>
</html>
"""
    
    with open(output_slider, 'w') as f:
        f.write(custom_html)
    
    print(f"Saved: {output_slider}")
    
    return output_slider


def create_detailed_table(mol_df, output_path):
    """Create a detailed table with all molecule information."""
    # Create a CSV with full side effect lists
    export_df = mol_df[['cid', 'name', 'pubchem_url', 'n_side_effects', 'smiles']].copy()
    export_df['side_effects_list'] = mol_df['side_effects'].apply(lambda x: '; '.join(x) if x else 'None')
    
    csv_path = output_path.replace('.html', '_data.csv')
    export_df.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")
    return csv_path


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
    
    # Load checkpoint to get max_side_effects parameter
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_args = checkpoint['args']
    max_side_effects = checkpoint_args.get('max_side_effects', None) if isinstance(checkpoint_args, dict) else getattr(checkpoint_args, 'max_side_effects', None)
    
    if max_side_effects:
        print(f"Model trained on top {max_side_effects} side effects")
    
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
    
    # Load molecule information (filtered to match model training)
    mol_df = load_molecule_info(cids, max_side_effects=max_side_effects)
    
    # Create interactive plot
    output_path = os.path.join(args.output_dir, 'umap_interactive.html')
    html_path = create_interactive_plot(umap_embeddings, mol_df, output_path)
    
    # Create detailed table
    csv_path = create_detailed_table(mol_df, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Visualization Done")
    print(f"{'='*60}")
    print(f"Interactive plot: {html_path}")
    print(f"Data export: {csv_path}")
    print(f"\n To view: open {html_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
