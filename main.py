import torch
from utils import MyDataLoader, train_generator, train_discriminator
from model import GraphGenerator, Discriminator
import numpy as np
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
# generator_path = 'model.001/generator.pt'
# discriminator_path = 'model.001/discriminator.pt'

dataloader = MyDataLoader(filepath='../../tox21.csv')
df = dataloader.load()
adj, feat, atom_mapping, bond_mapping = dataloader.preprocess(df, 10, 'smiles')

BATCH_SIZE = 32
EPOCHS = 300
DROPOUT_RATE = 0.2
LATENT_DIM = 64
BOND_DIM = adj.shape[1]
NUM_ATOMS = adj.shape[-1]
ATOM_DIM = feat.shape[-1]

generator = GraphGenerator(
    BATCH_SIZE, DROPOUT_RATE, LATENT_DIM, [128, 256, 512], BOND_DIM, NUM_ATOMS, ATOM_DIM
).to(device)

discriminator = Discriminator(
    [(128, 64), 256], [256, 1], ATOM_DIM, BOND_DIM, DROPOUT_RATE
).to(device)

optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

adj_tensor, feat_tensor = dataloader.train_batch(adj, feat, BATCH_SIZE)
def main():
    indices = torch.arange(0, adj_tensor.shape[0], 1)
    disc_steps = 5
    history = np.zeros((EPOCHS, 2))
    for epoch in tqdm(range(EPOCHS)):
        gen_running = 0
        disc_running = 0
        for idx in indices:
            graph_real = [adj_tensor[idx].to(device), feat_tensor[idx].to(device)]
            for _ in range(disc_steps):
                z = torch.normal(mean=0, std=1, size=(BATCH_SIZE, LATENT_DIM), device=device)
                d_loss = train_discriminator(graph_real, generator, discriminator, z, BATCH_SIZE, optimizer_disc)
                disc_running += d_loss

            z = torch.normal(mean=0, std=1, size=(BATCH_SIZE, LATENT_DIM), device=device)
            g_loss = train_generator(generator, discriminator, z, optimizer_gen)
            gen_running += g_loss

        avg_gen = gen_running / len(indices)
        avg_disc = (disc_running / disc_steps) / len(indices)
        history[epoch] = avg_gen, avg_disc
        indices = dataloader.shuffle_index(len(indices))

        if (epoch % 10 == 0) or (epoch == EPOCHS - 1):
            # save_generator(epoch, generator_path, generator, optimizer_gen, g_loss)
            # save_discriminator(epoch, discriminator_path, discriminator, optimizer_disc, d_loss)
            # image_path = f'model.001/images/epoch.{epoch}'
            # save_images(generator, BATCH_SIZE, LATENT_DIM, image_path, atom_mapping, bond_mapping)
            print(f'Discriminator Loss: {avg_disc}, Generator Loss: {avg_gen}')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(history[:, 0], '-', label='Generator')
    ax.plot(history[:, 1], '-', label='Discriminator')
    ax.set(
        xlabel='Epochs',
        ylabel='Loss',
        xlim=[0, EPOCHS],
        ylim=[history[:, 1].min(), history[:, 1].max()]
    )
    plt.savefig(f'fig.history.{10}.{EPOCHS}.pdf')
if __name__ == "__main__":
    main()
