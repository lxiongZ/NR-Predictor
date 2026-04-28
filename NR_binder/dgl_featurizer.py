import numpy as np
import torch
from rdkit import Chem

from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

class DGLFeaturizer:
    def __init__(self, use_edges=True, use_chirality=True, add_self_loop=False):
        self.use_edges = use_edges
        self.use_chirality = use_chirality
        self.add_self_loop = add_self_loop

        self.node_featurizer = CanonicalAtomFeaturizer(atom_data_field='atomic')
        self.edge_featurizer = CanonicalBondFeaturizer(
            bond_data_field='atomic',
            self_loop=add_self_loop
        )
    
    def featurize(self, mol):
        if mol is None:
            raise ValueError("Invalid molecule")

        node_features = self.extract_node_features(mol)

        edge_index, edge_features = self.extract_edge_features(mol)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features
        }
    

    def extract_node_features(self, mol):
        node_features = self.node_featurizer(mol)['atomic']
        return node_features.numpy() if hasattr(node_features, 'numpy') else node_features

    # def extract_edge_features(self, mol):
    #     """
    #     """
    #     num_atoms = mol.GetNumAtoms()

    #     edge_indices = []
    #     edge_features = []

    #     if self.use_edges:
    #         try:
    #             mol_edge_features = self.edge_featurizer(mol)['atomic']
    #             mol_edge_features_np = mol_edge_features.numpy() if hasattr(mol_edge_features, 'numpy') else mol_edge_features
    #         except:
    #             mol_edge_features_np = None

    #     bond_idx = 0
    #     """
    #     """
    #         j = bond.GetEndAtomIdx()

    #         edge_indices.extend([[i, j], [j, i]])

    #         if self.use_edges:
    #             if mol_edge_features_np is not None and bond_idx < len(mol_edge_features_np):
    #                 bond_feat = mol_edge_features_np[bond_idx]
                
    #             else:
    #                 bond_type = bond.GetBondType()
    #                 simple_feat = [float(bond_type)]
    #                 edge_features.extend([simple_feat, simple_feat])

    #         bond_idx += 1

    #     if self.add_self_loop:
    #         for i in range(num_atoms):
    #             edge_indices.append([i, i])
    #             if self.use_edges and edge_features:
    #                 self_loop_feat = np.zeros_like(edge_features[0])
    #                 edge_features.append(self_loop_feat)

    #     """
    #     """
    #     if edge_indices:
    #         edge_index = np.array(edge_indices).T  # [2, num_edges]
    #     else:
    #         edge_index = np.array([[], []], dtype=np.int64)

    #     if edge_features:
    #         edge_attr = np.array(edge_features, dtype=np.float32)
    #     else:
    #         num_edges = len(edge_indices)
    #         edge_attr = np.ones((num_edges, 1), dtype=np.float32) if num_edges > 0 else np.array([]).reshape(0, 1)

    #     return edge_index, edge_attr

    def extract_edge_features(self, mol):

        num_atoms = mol.GetNumAtoms()
        # num_bonds = mol.GetNumBonds()
        
        edge_attr = self.edge_featurizer(mol)['atomic'].numpy()
        
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([[i, j], [j, i]])
        
        if self.add_self_loop:
            edges.extend([[i, i] for i in range(num_atoms)])

        edge_index = np.array(edges).T
        
        return edge_index, edge_attr


