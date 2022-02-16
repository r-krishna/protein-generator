"""
This file contains methods that take atom and bond objects and calculate different features from them
"""
import numpy as np

from protein_generator.datasets.utils import ATOMTYPES, BONDTYPES

def one_hot_encode_unk(x, Xs):
	"""
	One hot encodes a value in a list. If the value isn't in the list, the last encoding is "other"
	
	Args:
		x: the item that is to be encoded
		Xs: a list of items that are to be encoded

	Returns:
		List: a list of length len(Xs) + 1 that is a one hot encoding of the features in Xs
	"""
	ret = np.zeros(len(Xs) + 1)
	if x not in Xs:
		ret[-1] = 1.0
	else:
		ndx = Xs.index(x)
		ret[ndx] = 1.0
	return list(ret)


def get_atom_types(atoms):
	"""
	Takes a list of Atom objects (defined in protein_generator.datasets.utils.Atom) and computes the one hot encoding 
	of the atom types
	"""
	atom_features = [one_hot_encode_unk(atom, ATOMTYPES) for atom in atoms]
	return atom_features


def get_bond_types(bonds):
	"""
	Takes a list of Bond objects (defined in protein_generator.datasets.utils.Bond) and computes the one hot encoding 
	of the atom types
	"""
	bond_features = [one_hot_encode_unk(bond, BONDTYPES) for bond in bonds]
	return bond_features

