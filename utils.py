import torch
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage, MolToImageFile
from smiles import smiles_to_graph, graph_to_molecule

class MyDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        return pd.read_csv(self.filepath)  

    def preprocess(self, dataframe, threshold, column='smiles'):
        smiles = []
        atoms = {}
        for idx, smile in enumerate(dataframe[column]):
            molecule = Chem.MolFromSmiles(smile)
            n_atoms = molecule.GetNumAtoms()
            if n_atoms <= threshold:
                smiles.append(idx)
                for atom in molecule.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in atoms:
                        atoms[symbol] += 1
                    else:
                        atoms[symbol] = 1

        reduced = dataframe.iloc[smiles]

        atom_mapping = {sym: i for i, sym in enumerate(atoms.keys())}
        atom_type = {i: sym for i, sym in enumerate(atoms.keys())}
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

        self.NUM_ATOMS = threshold
        self.ATOM_DIM = len(atom_type) + 1
        self.BOND_DIM = 5

        adjacency, features = [], []
        for smile in reduced['smiles']:
            x, y = smiles_to_graph(smile, atom_mapping, bond_mapping, self.BOND_DIM, self.NUM_ATOMS, self.ATOM_DIM)
            adjacency.append(x)
            features.append(y)
        return np.array(adjacency), np.array(features), atom_type, bond_mapping
    
    def train_batch(self, adjacency, features, batch_size):
        N = adjacency.shape[0] // batch_size
        index = N * batch_size
        adjacency, features = adjacency[:index], features[:index]
        adjacency_tensor = torch.tensor(adjacency)
        features_tensor = torch.tensor(features)
        return adjacency_tensor.view(N, batch_size, self.BOND_DIM, self.NUM_ATOMS, self.NUM_ATOMS), features_tensor.view(N, batch_size, self.NUM_ATOMS, self.ATOM_DIM)
    
    def shuffle_index(self, length):
        indices = torch.randperm(length)
        return indices

def gradient_penalty(graph_real, graph_fake, batch_size, discriminator):
    alpha = torch.rand(size=(batch_size, )).view(batch_size, 1, 1, 1)
    adjacency_real, feature_real = graph_real
    adjacency_fake, feature_fake = graph_fake
    adjacency_interp = ((adjacency_real * alpha) + (1 - alpha) * adjacency_fake).requires_grad_(True)
    alpha = alpha.view(batch_size, 1, 1)
    features_interp = ((feature_real * alpha) + (1 - alpha) * feature_fake).requires_grad_(True)

    graph_interp = [adjacency_interp, features_interp]
    output = discriminator(graph_interp)
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=graph_interp,
        grad_outputs=torch.ones(output.size()),
        create_graph=True, retain_graph=True
    )

    adj_penalty = (1 - torch.linalg.norm(gradients[0], dim=1)) ** 2
    feat_penalty = (1 - torch.linalg.norm(gradients[1], dim=2)) ** 2
    return torch.mean(adj_penalty.mean(dim=(-2, -1)) + feat_penalty.mean(dim=-1))

def disc_loss(real_logits, fake_logits, penalty):
    loss = -(torch.mean(real_logits) - torch.mean(fake_logits))
    return loss + penalty * 10
def gen_loss(fake_logits):
    return -torch.mean(fake_logits)

def train_discriminator(graph_real, generator, discriminator, z, batch_size):
    with torch.no_grad():
        graph_fake = generator(z)
    real_logits = discriminator(graph_real)
    fake_logits = discriminator(graph_fake)
    penalty = gradient_penalty(graph_real, graph_fake, batch_size, discriminator)
    d_loss = disc_loss(real_logits, fake_logits, penalty)
    return d_loss

def train_generator(generator, discriminator, z):
    graph_fake = generator(z)
    fake_logits = discriminator(graph_fake)
    g_loss = gen_loss(fake_logits)
    return g_loss

def save_discriminator(epoch, path, model, optimizer, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def save_generator(epoch, path, model, optimizer, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def sample(generator, batch_size, latent_dim, atom_mapping, bond_mapping):
    z = torch.normal(0, 1, size=(batch_size, latent_dim))
    with torch.no_grad():
        graph = generator(z)
    bond_dim = graph[0].shape[1]
    atom_dim = graph[1].shape[-1]

    adjacency = torch.argmax(graph[0], dim=1)
    adjacency = torch.nn.functional.one_hot(adjacency, num_classes=bond_dim).permute(0, 3, 1, 2)
    features = torch.argmax(graph[1], dim=2)
    features = torch.nn.functional.one_hot(features, num_classes=atom_dim)

    return [
        graph_to_molecule([adjacency[i].numpy(), features[i].numpy()], atom_mapping, bond_mapping, bond_dim, atom_dim)
        for i in range(batch_size)
    ]

def save_images(generator, batch_size, latent_dim, path, atom_mapping, bond_mapping):
    molecules = sample(generator, batch_size, latent_dim, atom_mapping, bond_mapping)
    for idx, mol in enumerate(molecules):
        if mol is not None:
            MolToImageFile(mol, filename=f'{path}.{idx}.png', size=(150, 150))