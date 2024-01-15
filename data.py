import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import numpy as np
from smiles import smiles_to_graph, graph_to_molecule

df = pd.read_csv('../tox21.csv')

heavy_atoms = {}
for smile in df['smiles']:
    molecule = Chem.MolFromSmiles(smile)
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in heavy_atoms:
            heavy_atoms[symbol] += 1
        else:
            heavy_atoms[symbol] = 1

max_heavy_atoms = -1e8
for smile in df['smiles']:
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = len(molecule.GetAtoms())
    if n_atoms > max_heavy_atoms:
        max_heavy_atoms = n_atoms

atom_mapping = {sym: i for i, sym in enumerate(heavy_atoms.keys())}
atom_type = {i: sym for i, sym in enumerate(heavy_atoms.keys())}
bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}

NUM_ATOMS = max_heavy_atoms
ATOM_DIM = len(atom_type)
BOND_DIM = 4 + 1

# smile = reduced['smiles'].iloc[1000]

# # print(smile)
# adjacency, features = smiles_to_graph(smile, atom_mapping, bond_mapping, BOND_DIM, NUM_ATOMS, ATOM_DIM)
# print(adjacency, features)

# graph = [adjacency, features]
# molecule = graph_to_molecule(graph, atom_type, bond_mapping, BOND_DIM, ATOM_DIM)
# width, height = 300, 300
# Chem.Draw.MolToFile(molecule, 'molecule.png')
