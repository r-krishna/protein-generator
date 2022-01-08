import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType


ATOMTYPES = [
    "H",
    # "Li",
    # "Be",
    # "B",
    "C",
    "N",
    "O",
    # "F",
    # "Na",
    # "Mg",
    # "Al",
    # "Mn",
    # "Si",
    # "P",
    "S",
    # "Cl",
    # "K",
    # "Ca",
    # "Fe",
    # "Zn",
    # "Br",
    # "I",
]  # 22, 23 with unk

BONDTYPES = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]  # 4, 5 with unk

aa2atoms = {
    "ALA": ("N", "CA", "C", "O", "CB"),  # ala
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),  # arg
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),  # asn
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),  # asp
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),  # cys
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),  # gln
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),  # glu
    "GLY": ("N", "CA", "C", "O"),  # gly
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),  # his
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),  # ile
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),  # leu
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),  # lys
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),  # met
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"),  # phe
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),  # pro
    "SER": ("N", "CA", "C", "O", "CB", "OG"),  # ser
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),  # thr
    "TRP": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "NE1",
        "CZ2",
        "CZ3",
        "CH2",
    ),  # trp
    "TYR": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
    ),  # tyr
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),  # val
}

aa2graph = {
    "ALA": {"src": (0, 1, 1, 2), "dst": (1, 2, 4, 3), "btypes": (0, 0, 0, 2)},
    "ARG": {
        "src": (0, 1, 1, 2, 4, 5, 6, 7, 8, 8),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8, 9, 10),
        "btypes": (0, 0, 0, 2, 0, 0, 0, 0, 2, 0),
    },
    "ASN": {
        "src": (0, 1, 1, 2, 4, 5, 5),
        "dst": (1, 2, 4, 3, 5, 6, 7),
        "btypes": (0, 0, 0, 2, 0, 2, 0),
    },
    "ASP": {
        "src": (0, 1, 1, 2, 4, 5, 5),
        "dst": (1, 2, 4, 3, 5, 6, 7),
        "btypes": (0, 0, 0, 2, 0, 2, 0),
    },
    "CYS": {"src": (0, 1, 1, 2, 4), "dst": (1, 2, 4, 3, 5), "btypes": (0, 0, 0, 2, 0)},
    "GLN": {
        "src": (0, 1, 1, 2, 4, 5, 6, 6),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8),
        "btypes": (0, 0, 0, 2, 0, 0, 2, 0),
    },
    "GLU": {
        "src": (0, 1, 1, 2, 4, 5, 6, 6),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8),
        "btypes": (0, 0, 0, 2, 0, 0, 2, 0),
    },
    "GLY": {"src": (0, 1, 2), "dst": (1, 2, 3), "btypes": (0, 0, 2)},
    "HIS": {
        "src": (0, 1, 1, 2, 4, 5, 5, 6, 8, 9),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8, 9, 7),
        "btypes": (0, 0, 0, 2, 0, 4, 4, 4, 4, 4),
    },
    "ILE": {
        "src": (0, 1, 1, 2, 4, 4, 5),
        "dst": (1, 2, 4, 3, 5, 6, 7),
        "btypes": (0, 0, 0, 2, 0, 0, 0),
    },
    "LEU": {
        "src": (0, 1, 1, 2, 4, 5, 5),
        "dst": (1, 2, 4, 3, 5, 6, 7),
        "btypes": (0, 0, 0, 2, 0, 0, 0),
    },
    "LYS": {
        "src": (0, 1, 1, 2, 4, 5, 6, 7),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8),
        "btypes": (0, 0, 0, 2, 0, 0, 0, 0),
    },
    "MET": {
        "src": (0, 1, 1, 2, 4, 5, 6),
        "dst": (1, 2, 4, 3, 5, 6, 7),
        "btypes": (0, 0, 0, 2, 0, 0, 0),
    },
    "PHE": {
        "src": (0, 1, 1, 2, 4, 5, 6, 7, 8, 9, 10),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 5),
        "btypes": (0, 0, 0, 2, 0, 4, 4, 4, 4, 4, 4),
    },
    "PRO": {
        "src": (0, 1, 1, 2, 4, 5, 6),
        "dst": (1, 2, 4, 3, 5, 6, 0),
        "btypes": (0, 0, 0, 2, 1, 1, 1),
    },
    "SER": {"src": (0, 1, 1, 2, 4), "dst": (1, 2, 4, 3, 5), "btypes": (0, 0, 0, 2, 0)},
    "THR": {
        "src": (0, 1, 1, 2, 4, 4),
        "dst": (1, 2, 4, 3, 5, 6),
        "btypes": (0, 0, 0, 2, 0, 0),
    },
    "TRP": {
        "src": (0, 1, 1, 2, 4, 5, 5, 6, 7, 7, 8, 8, 9, 11, 12),
        "dst": (1, 2, 4, 3, 5, 6, 7, 10, 8, 9, 10, 11, 12, 13, 13),
        "btypes": (0, 0, 0, 2, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
    },
    "TYR": {
        "src": (0, 1, 1, 2, 4, 5, 5, 6, 7, 8, 9, 10),
        "dst": (1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 10, 11),
        "btypes": (0, 0, 0, 2, 0, 4, 4, 4, 4, 4, 4, 0),
    },
    "VAL": {
        "src": (0, 1, 1, 2, 4, 4),
        "dst": (1, 2, 4, 3, 5, 6),
        "btypes": (0, 0, 0, 2, 0, 0),
    },
}

def protein_atomname_to_atomobj(atomname, restype):
    """
    Converts protein atoms into Atom objects with hybridization, aromatic and ring features
    These atom objects can be used to make features for the ML inputs downstream

    Args:
        atomname (str): The atom name in the pdb file (eg. CA)
        restype (str): the residue type in the pdb file (eg. PRO or ALA)

    Returns:
        [se3_free_energy.utils.data_utils.Atom]: an Atom type 
    """
    # First backbone atom is almost the same for all restypes
    if atomname in ["N", "CA", "CB"]:
        if restype == "PRO":
            atom = Atom(atomname)
            atom.SetHybridization("SP3")
            atom.SetIsAromatic(False)
            atom.SetIsInRing(True)
        else:
            atom = sp3atom(atomname)

        return atom

    if atomname in ["C", "O"]:
        atom = Atom(atomname)
        atom.SetHybridization("SP2")
        atom.SetIsAromatic(False)
        atom.SetIsInRing(False)
        return atom

    # GLY and ALA shouldn't reach here
    assert restype not in ["GLY", "ALA"]

    # Only side chain atoms reach here except CB.
    if restype == "ARG":
        if atomname in ["CZ", "NH1"]:
            atom = Atom(atomname)
            atom.SetHybridization("SP2")
            atom.SetIsAromatic(False)
            atom.SetIsInRing(False)
        else:
            atom = sp3atom(atomname)

    elif restype == "ASN":
        if atomname in ["CG", "OD1"]:
            atom = Atom(atomname)
            atom.SetHybridization("SP2")
            atom.SetIsAromatic(False)
            atom.SetIsInRing(False)
        else:
            atom = sp3atom(atomname)

    elif restype == "ASP":
        if atomname in ["CG", "OD1"]:
            atom = Atom(atomname)
            atom.SetHybridization("SP2")
            atom.SetIsAromatic(False)
            atom.SetIsInRing(False)
        else:
            atom = sp3atom(atomname)

    elif restype == "GLN":
        if atomname in ["CD", "OE1"]:
            atom = Atom(atomname)
            atom.SetHybridization("SP2")
            atom.SetIsAromatic(False)
            atom.SetIsInRing(False)
        else:
            atom = sp3atom(atomname)

    elif restype == "GLU":
        if atomname in ["CD", "OE1"]:
            atom = Atom(atomname)
            atom.SetHybridization("SP2")
            atom.SetIsAromatic(False)
            atom.SetIsInRing(False)
        else:
            atom = sp3atom(atomname)

    elif restype in ["CYS", "LEU", "ILE", "VAL", "LYS", "SER", "THR", "MET"]:
        atom = sp3atom(atomname)

    elif restype in ["HIS", "PHE", "TRP"]:
        atom = Atom(atomname)
        atom.SetHybridization("SP2")
        atom.SetIsAromatic(True)
        atom.SetIsInRing(True)

    elif restype == "TYR":
        if atomname == "OH":
            atom = sp3atom(atomname)
        else:
            atom = Atom(atomname)
            atom.SetHybridization("SP2")
            atom.SetIsAromatic(True)
            atom.SetIsInRing(True)

    elif restype == "PRO":
        atom = Atom(atomname)
        atom.SetHybridization("SP3")
        atom.SetIsAromatic(False)
        atom.SetIsInRing(True)

    return atom


def sample_cube(radius=10):
    """
    Samples points in a cube with a given radius. This can be used to randomly initialize coordinates 
    """
    pass


class Atom:
    def __init__(self, atomname):
        self.element = atomname[0]
        self.hyb = "UNK"
        self.isaromatic = False
        self.isinring = False
        self.chiraltag = "UNK"
        self.total_Hs = 0
        self.charge = 0

    def SetSymbol(self, setting):
        self.element = setting

    def GetSymbol(self):
        return self.element

    def SetHybridization(self, setting):
        self.hyb = setting

    def GetHybridization(self):
        return self.hyb

    def SetIsAromatic(self, setting):
        self.isaromatic = setting

    def IsAromatic(self):
        return self.isaromatic

    def SetIsInRing(self, setting):
        self.isinring = setting

    def IsInRing(self):
        return self.isinring

    def SetChiralTag(self, setting):
        self.chiraltag = setting

    def GetChiralTag(self):
        return self.chiraltag

    def SetTotalNumHs(self, setting):
        self.total_Hs = setting

    def GetTotalNumHs(self):
        return self.total_Hs

    def SetCharge(self, setting):
        self.charge = setting

    def GetCharge(self):
        return self.charge


class Bond:
    def __init__(self, bondtype="UNK", inring=False):
        self.inring = inring
        self.bondtype = bondtype

    def SetIsInRing(self, setting):
        self.inring = setting

    def IsInRing(self):
        return self.inring

    def SetBondType(self, setting):
        self.bondtype = setting

    def GetBondType(self):
        return self.bondtype
