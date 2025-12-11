import pandas as pd
import os
import torch
from torch.utils.data import Dataset, random_split
import networkx as nx
import pickle
from torch_geometric.data import HeteroData
import numpy as np


class GraphDataset(Dataset):
    """
    Dataset for molecular graphs with typed edges.
    Converts NetworkX graphs to PyG HeteroData with bond types as edge types.
    """

    def __init__(self, graph_dir, cid_se_csv=None, task_config=None, feature_key='x', max_side_effects=None):
        super().__init__()
        self.graph_dir = graph_dir
        self.feature_key = feature_key

        # Task configuration
        self.task_configs = {}
        if cid_se_csv:
            self.task_configs['side_effects'] = cid_se_csv
        if task_config:
            self.task_configs.update(task_config)

        self.task_cid_maps = {}
        self.task_dims = {}
        self.has_other_ses_map = {}  # Track drugs with non-top-N side effects
        self.task_stats = {}  # Store mean/std for regression tasks

        for task_name, csv_path in self.task_configs.items():
            df = pd.read_csv(csv_path).set_index("cid")

            # For side_effects, optionally filter to top N most common
            if task_name == 'side_effects' and max_side_effects is not None:
                # Select top N side effects by prevalence (sum of occurrences)
                col_sums = df.sum(axis=0).sort_values(ascending=False)
                top_cols = col_sums.head(max_side_effects).index.tolist()
                
                # Before filtering, identify drugs with side effects NOT in top N
                for cid, row in df.iterrows():
                    total_ses = row.sum()  # Total number of side effects
                    top_n_ses = row[top_cols].sum()  # Number of top-N side effects
                    has_other = (total_ses > top_n_ses)  # Has SEs not in top N
                    self.has_other_ses_map[int(cid)] = float(has_other)
                
                # Now filter to top N
                df = df[top_cols]
                print(f"Limited side_effects to top {max_side_effects} (from {len(col_sums)} total)")
                print(f"  {sum(self.has_other_ses_map.values())} drugs have side effects not in top {max_side_effects}")

            # Normalize regression tasks (molprops)
            if task_name == 'molprops':
                # Compute mean and std for normalization
                mean = df.mean(axis=0).values
                std = df.std(axis=0).values
                std[std == 0] = 1.0  # Avoid division by zero
                
                self.task_stats[task_name] = {
                    'mean': torch.tensor(mean, dtype=torch.float32),
                    'std': torch.tensor(std, dtype=torch.float32)
                }
                
                # Normalize the dataframe
                df = (df - mean) / std
                print(f"Normalized {task_name}: mean={mean.round(2)}, std={std.round(2)}")
            
            # Store dimensions for model initialization
            # Add +1 for "has_other_SEs" flag if we filtered side effects
            if task_name == 'side_effects' and max_side_effects is not None:
                self.task_dims[task_name] = len(df.columns) + 1  # +1 for has_other_SEs
            else:
                self.task_dims[task_name] = len(df.columns)

            cid_map = {
                int(cid): torch.tensor(row.values, dtype=torch.float32)
                for cid, row in df.iterrows()
            }
            self.task_cid_maps[task_name] = cid_map

            if task_name == 'side_effects':
                self.se_cols = list(df.columns)

        # Load graph files
        files = os.listdir(graph_dir)
        items = []
        for file in files:
            if file.endswith('.gpickle'):
                cid = int(file.split(".")[0])
                full_file = os.path.join(graph_dir, file)
                items.append((cid, full_file))

        self.items = sorted(items, key=lambda t: t[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        cid, full_file = self.items[idx]
        with open(full_file, "rb") as f:
            G = pickle.load(f)

        # Convert to HeteroData
        data = self._networkx_to_hetero(G)
        
        data.cid = torch.tensor([cid], dtype=torch.int64)
        
        # Attach targets for each task with masking
        for task_name, cid_map in self.task_cid_maps.items():
            if cid in cid_map:
                target = cid_map[cid].unsqueeze(0)
                
                # For side_effects with filtering, append the "has_other_SEs" flag
                if task_name == 'side_effects' and hasattr(self, 'has_other_ses_map') and len(self.has_other_ses_map) > 0:
                    has_other = torch.tensor([[self.has_other_ses_map.get(cid, 0.0)]], dtype=torch.float32)
                    target = torch.cat([target, has_other], dim=1)
                
                setattr(data, f"y_{task_name}", target)
                setattr(data, f"mask_{task_name}", torch.tensor([True], dtype=torch.bool))
                
                if task_name == 'side_effects':
                    data.y = target
            else:
                dim = self.task_dims[task_name]
                dummy = torch.zeros((1, dim), dtype=torch.float32)
                setattr(data, f"y_{task_name}", dummy)
                setattr(data, f"mask_{task_name}", torch.tensor([False], dtype=torch.bool))
                
                if task_name == 'side_effects':
                    data.y = dummy
                    
        return data

    def _networkx_to_hetero(self, G: nx.Graph) -> HeteroData:
        """
        Convert NetworkX graph to PyG HeteroData with heterogeneous edges.
        Node type: 'atom'
        Edge types: ('atom', 'SINGLE', 'atom'), ('atom', 'DOUBLE', 'atom'), etc.
        """
        data = HeteroData()
        
        # Extract node features
        num_nodes = G.number_of_nodes()
        node_mapping = {n: i for i, n in enumerate(G.nodes())}
        
        if num_nodes == 0:
            # Empty graph
            data['atom'].x = torch.zeros((0, 154), dtype=torch.float32)
            return data
        
        # Get node features
        first_node = list(G.nodes())[0]
        if self.feature_key in G.nodes[first_node]:
            features = []
            for n in G.nodes():
                feat = G.nodes[n][self.feature_key]
                features.append(np.array(feat))
            data['atom'].x = torch.tensor(np.stack(features), dtype=torch.float32)
        else:
            # Fallback to ones
            feat_dim = 154  # Default from graphs_v2
            data['atom'].x = torch.ones((num_nodes, feat_dim), dtype=torch.float32)
        
        # Group edges by bond type
        edge_dict = {}
        for u, v, edge_data in G.edges(data=True):
            bond_type = edge_data.get('bond_type', 'SINGLE')
            
            if bond_type not in edge_dict:
                edge_dict[bond_type] = []
            
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            
            # Add both directions for undirected graph
            edge_dict[bond_type].append([u_idx, v_idx])
            edge_dict[bond_type].append([v_idx, u_idx])
        
        # Create edge_index for each bond type
        for bond_type, edges in edge_dict.items():
            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data['atom', bond_type, 'atom'].edge_index = edge_index
        
        return data


def make_splits(dataset, train=0.8, val=0.1, seed=42):
    n = len(dataset)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)
