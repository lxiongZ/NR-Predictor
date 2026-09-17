import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
from tqdm import tqdm
from rdkit import Chem
from dgl_featurizer import DGLFeaturizer

class LoadDataset(InMemoryDataset):
    def __init__(self, root, raw_filename, transform=None, pre_transform=None):
        self.raw_filename = raw_filename
        super(LoadDataset, self).__init__(root, transform, pre_transform)

        self.all_data = torch.load(self.processed_paths[0])

        self.train_idx = []
        self.val_idx = []
        self.test_idx = []

        for i in range(len(self.all_data)):
            split = self.all_data[i].split
            if split == 'train':
                self.train_idx.append(i)
            elif split == 'val':
                self.val_idx.append(i)
            elif split == 'test':
                self.test_idx.append(i)

    @property
    def raw_file_names(self):
        return self.raw_filename

    @property
    def processed_file_names(self):
        return ['all_molecules.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        featurizer = DGLFeaturizer(use_edges=True, use_chirality=True, add_self_loop=False)
        all_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
            smiles = row.iloc[0]
            mol = Chem.MolFromSmiles(smiles)

            # if mol is None:
            #     print(f"Warning: Invalid SMILES at index {idx}: {smiles}")
            #     continue

            f = featurizer.featurize(mol)
            node_features = torch.tensor(f['node_features'])
            edge_index = torch.tensor(f['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(f['edge_features'])

            num_edges = edge_index.size(1)
            rev_edge_index = torch.arange(num_edges, dtype=torch.long)
            rev_edge_index[::2] = torch.arange(1, num_edges, 2)
            rev_edge_index[1::2] = torch.arange(0, num_edges, 2)

            label_cols = df.columns[1:-2]
            labels = torch.tensor(row[label_cols].astype(float).values, dtype=torch.float32)
            labels = labels.unsqueeze(0)

            source = row.iloc[-2]

            split = row.iloc[-1]

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                rev_edge_index=rev_edge_index,
                y=labels,
                split=split,
                source=source,
                smiles=smiles
            )
            all_data.append(data)

        print(f"保存 {len(all_data)} 个分子到 {self.processed_paths[0]}")
        torch.save(all_data, self.processed_paths[0])

    def len(self):
        return len(self.all_data)

    def get(self, idx):
        return self.all_data[idx]

    def get_split_dataset(self, split='train'):
        if split == 'train':
            return [self.all_data[i] for i in self.train_idx]
        elif split == 'val':
            return [self.all_data[i] for i in self.val_idx]
        elif split == 'test':
            return [self.all_data[i] for i in self.test_idx]
        else:
            raise ValueError(f"Unknown split: {split}")

def mol_to_graph_data_obj_simple(mol):
    from dgl_featurizer import DGLFeaturizer
    import torch
    from torch_geometric.data import Data

    featurizer = DGLFeaturizer(use_edges=True, use_chirality=True, add_self_loop=False)

    f = featurizer.featurize(mol)

    node_features = torch.tensor(f['node_features'])
    edge_index = torch.tensor(f['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(f['edge_features'])

    num_edges = edge_index.size(1)
    rev_edge_index = torch.arange(num_edges, dtype=torch.long)
    rev_edge_index[::2] = torch.arange(1, num_edges, 2)
    rev_edge_index[1::2] = torch.arange(0, num_edges, 2)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
               rev_edge_index=rev_edge_index)

    data.smiles = Chem.MolToSmiles(mol)

    return data
