import os

from Bio.PDB import PDBParser

from protein_generator.datasets.base import DataModule
from protein_generator.datasets.graph import PDBChainData, RandomNoiseData


class ProteinChainDataset(DataModule):
    
    def __init__(self, pdb_dir, pdb_list) -> None:
        super().__init__()
        self.pdb_dir = pdb_dir
        self.pdb_list = list(open(pdb_list, "r+"))
        self.parser = PDBParser(PERMISSIVE=1)

    def __getitem__(self, i):
        pdb_id = self.pdb_list[i]
        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        data = PDBChainData(self.parser, pdb_path)
        return data.build_graph()

    def __len__(self):
        return len(self.pdb_list)


class RandomNoiseDataset(DataModule):
    
    def __init__(self, num_samples, num_residues, num_sidechain_atoms, num_node_feats, num_edge_feats) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_residues = num_residues
        self.num_sidechain_atoms = num_sidechain_atoms
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

    def __getitem__(self, i):
        data = RandomNoiseData(self.num_residues, self.num_sidechain_atoms, self.num_node_feats, self.num_edge_feats)
        return data.build_graph()

    def __len__(self):
        return self.num_samples
