import itertools

import dgl
from dgl.data import DGLDataset

import numpy as np 
import torch

import rdkit
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType, ChiralType

from protein_generator.datasets.featurizers import get_atom_types, get_bond_types


class PDBChainData:
	""" 
	This class parses a chain of a pdb file and creates an undirected graph representation
	This class only considers residue atoms, and not ions. This is because ions usually aren't
	present at protein-protein interfaces
	"""
	
	def __init__(self, parser, pdb_file, chain_id):
		self.parser = parser
		self.pdb_file = pdb_file
		self.chain_id = chain_id

	def build_graph(self):
		"""
		constructs a graph for a single chain of a protein
		"""
		structure = self.parser.get_structure("structure", self.pdb_file)

		chain = [chain for chain in structure.get_chains() if chain.id == self.chain_id]

		if not chain:
			raise ValueError(f"chain {self.chain_id} was not found in the given PDB file, {self.pdb_file}")

		# chain is a list where the first value is the chain of interest
		chain = chain[0]

		residues = chain.get_residues()

		protein_atoms, restypes, protein_coords = self.convert_protein_to_atoms(residues)
		atom_features = get_atom_types(protein_atoms)
		protein_src, protein_dst, protein_bonds, num_protein_atoms = self.get_protein_bonds(residues)
		bond_features = get_bond_types(protein_bonds)

		assert len(protein_atoms) == num_protein_atoms, "Numbering of nodes for bond features doesn't match node features"
		assert len(protein_atoms) == len(protein_coords), "Every protein atom does not have a corresponding coordinate"
		src, dst = protein_src.to(dtype=torch.int64), protein_dst.to(dtype=torch.int64)
		G = dgl.graph((src, dst))
		atom_features = torch.tensor(atom_features)
		bond_features = torch.tensor(bond_features)
		return G, atom_features, bond_features

	@staticmethod
	def convert_protein_to_atoms(residues):
		"""
		Converts atoms within a set of residues parsed by Bio.PDB.PDBParser into an Atom wrapper class with different features of atoms
		Also returns what residue each atom belonged to for future featurization

		Args:
			residues (List[Bio.PDB.Residue]): a list of biopython residues that are a part of the input graph 

		Returns:
			List[se3_free_energy.utils.data_utils.Atom]: a list of Atom objects
			List[str]: list of restypes for each atom in the protein
			List[np.array]: list of np arrays (3,) of coordinates of each atom
		"""
		atoms = []
		restypes = []
		coords = []
		for residue in residues:
			restype = residue.get_resname()
			if restype in aa2atoms:
				atom_names = aa2atoms[restype]
				for atom_name in atom_names:
					atom_obj = protein_atomname_to_atomobj(atom_name, restype)
					atoms.append(atom_obj)
					restypes.append(restype)
					coords.append(residue.child_dict[atom_name].get_coord())
		return atoms, restypes, coords

	@staticmethod
	def get_protein_bonds(residues):
		"""
		Create edges and bond features between all the atoms that are selected for creation of the graph

		Args:
			residues (List[Bio.PDB.Residue]): a list of selected residues for incorporation into the graph

		Returns:
			List[int]: a list of nodes (represented as their index- int) that are the starting point for edges
			List[int]: a list of nodes (represented as their index- int) that are the destinations of edges
			List[int]: a list of bond features (the options are captured in se3_free_energy.utils.util)
			int: the number of atoms that were selected to be in the graph

		"""
		# Keep track of the (previous_bb_C_atom, previous_residue_id)
		prev = (None, None)
		# Each atom is a node, so we want to count what node number each atom is when constructing the edges
		atom_counter = 0

		# set up the edges and edge features of the bonds
		src_all, dst_all, bonds = [], [], []
		for residue in residues:
			restype = residue.get_resname()
			curr_res_id = residue.get_id()[1]
			if restype in aa2atoms and restype in aa2graph:
				res_graph = aa2graph[restype]
				atoms = aa2atoms[restype]

				src = list(np.array(res_graph["src"]) + atom_counter)
				dst = list(np.array(res_graph["dst"]) + atom_counter)
				src_all.extend(src)
				dst_all.extend(dst)
				bonds.extend(res_graph["btypes"])
				if prev[0] and prev[1]:
					prev_C_idx, prev_res_id = prev
					if curr_res_id - prev_res_id == 1:
						# the backbone N is always the first atom in a residue
						curr_N_idx = atom_counter
						src_all.extend([prev_C_idx])
						dst_all.extend([curr_N_idx])
						# Treat the omega peptide bonds as double bonds
						bonds.extend([2])
				else:
					curr_N_idx = atom_counter
				# the backbone carbon is the third index for all residues
				curr_C_idx = curr_N_idx + 2
				prev = (curr_C_idx, curr_res_id)
				atom_counter += len(atoms)
		bonds = [peptide_btypes[bidx] for bidx in bonds]
		return src_all, dst_all, bonds, atom_counter


class RandomNoiseData:
	"""
	This class creates an input graph that has a connected protein backbone with randomly
	placed atoms. The beta carbons and the sidechain atoms are fully connected. The 
	network should learn what sidechains are connected to each backbone and the atom positions 
	of all the atoms
	"""
	def __init__(self, num_residues, num_sidechain_atoms, num_node_feats, num_edge_feats):
		self.num_residues = num_residues
		self.num_sidechain_atoms = num_sidechain_atoms
		self.num_node_feats = num_node_feats
		self.num_edge_feats = num_edge_feats

	def build_graph(self):
		"""
		Build a graph with a protein backbone and randomly placed side chain atoms
		"""
		backbone_atoms, backbone_restypes, backbone_coords = self.build_protein_backbone_atoms()
		backbone_src, backbone_dst, backbone_bonds, num_backbone_atoms = self.build_protein_backbone_bonds()
		backbone_atom_features = get_atom_types(backbone_atoms)
		backbone_bond_features = get_bond_types(backbone_bonds)
		assert (len(backbone_atoms) == num_backbone_atoms), "Backbone atoms are not correctly numbered"
		
		sidechain_atom_features, sidechain_coords = self.build_sidechain_atoms()
		sidechain_src, sidechain_dst, sidechain_bond_features = self.build_sidechain_bonds()

		src, dst, coords, atom_features, bond_features = [], [], [], [], []
		src.extend(backbone_src)
		src.extend(sidechain_src)

		dst.extend(backbone_dst)
		dst.extend(sidechain_dst)

		coords.extend(backbone_coords)
		coords.extend(sidechain_coords)

		atom_features.extend(backbone_atom_features)
		atom_features.extend(sidechain_atom_features)

		bond_features.extend(backbone_bond_features)
		bond_features.extend(sidechain_bond_features)

		pass

	def build_protein_backbone_atoms(self):
		"""
		Sets up atom objects for a set of glycine residues that form a protein backbone with coordinates
		placed at the origin
		"""
		atoms = []
		restypes = []
		coords = []
		for i in range(self.num_residues):
			restype = "GLY"
			atom_names = aa2atoms[restype]
			for atom_name in atom_names:
				atom_obj = protein_atomname_to_atomobj(atom_name, restype)
				atoms.append(atom_obj)
				restypes.append("UNK")
				coords.append(np.zeros(3))
		return atoms, restypes, coords

	def build_protein_backbone_bonds(self):
		"""
		Stitches together a set of glycines and returns the edges and bonds associated with the backbone
		"""
		# Keep track of the (previous_bb_C_atom, previous_residue_id)
		prev = (None, None)
		# Count the number of atoms added for correct node numbering
		atom_counter = 0
		# initialize the backbone as a set of glycines with length num_residues
		for i in range(self.num_residues):
			restype = "GLY"
			res_graph = aa2graph[restype]
			atoms = aa2atoms[restype]

			src = list(np.array(res_graph["src"]) + atom_counter)
			dst = list(np.array(res_graph["dst"]) + atom_counter)
			src_all.extend(src)
			dst_all.extend(dst)
			bonds.extend(res_graph["btypes"])
			if prev[0] and prev[1]:
				prev_C_idx, prev_res_id = prev
				if curr_res_id - prev_res_id == 1:
					# the backbone N is always the first atom in a residue
					curr_N_idx = atom_counter
					src_all.extend([prev_C_idx])
					dst_all.extend([curr_N_idx])
					# Treat the omega peptide bonds as double bonds
					bonds.extend([2])
			else:
				curr_N_idx = atom_counter
			# the backbone carbon is the third index for all residues
			curr_C_idx = curr_N_idx + 2
			prev = (curr_C_idx, curr_res_id)
			atom_counter += len(atoms)
		bonds = [peptide_btypes[bidx] for bidx in bonds]

		assert (atom_counter == self.num_residues * 4), "There are not 4 backbone atoms per residue"
		return src_all, dst_all, bonds, atom_counter 

	def build_sidechain_atoms(self):
		"""
		adds in random noise vectors for atom types and positions them all at the origin 
		"""
		node_feats = np.random.rand(self.num_sidechain_atoms, self.num_node_feats)
		coords = np.zeros(self.num_sidechain_atoms, 3)
		return node_feats, coords

	def build_sidechain_bonds(self):
		"""
		The sidechain atoms are all connected to eachother and all beta carbons. The network
		should learn which bonds are necessary and what types they are
		"""
		# beta carbons are the 3rd value in the glycine atom list (index=2). Each backbone has 
		# 4 values so the beta carbons are at 4n + 2
		beta_carbons = [4*n + 2 for n in range(self.num_residues)]
		num_backbone_atoms = self.num_residues * 4
		src, dst = [], []

		# connect the beta carbons to all the sidechain atoms
		for i in range(self.num_sidechain_atoms):
			src.extend(beta_carbons)
			dst.extend([i + num_backbone_atoms for i in range(len(beta_carbons))])

		# connect all the sidechain atoms to eachother
		for i, j in itertools.combinations(range(self.num_sidechain_atoms), 2):
			src.append(i + num_backbone_atoms)
			dst.append(j + num_backbone_atoms)

		bond_feats = np.random.rand(len(src), self.num_edge_feats)

		return src, dst, bond_feats



class ProteinInteractionDataset(DGLDataset):
	pass

