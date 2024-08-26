import torch
from utils import MyDataLoader, disc_loss, gen_loss, gradient_penalty, save_discriminator, save_generator, save_images
from model import GraphGenerator, Discriminator
from tqdm import tqdm

# Adding a line of text
generator_path = 'model.001/generator.pt'
discriminator_path = 'model.001/discriminator.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
# print(torch.cuda.is_available())
print(device)

dataloader = MyDataLoader(filepath='../../tox21.csv')
df = dataloader.load()
adj, feat, atom_mapping, bond_mapping = dataloader.preprocess(df, 10, 'smiles')

BATCH_SIZE = 32
EPOCHS = 200
DROPOUT_RATE = 0.2
LATENT_DIM = 64
BOND_DIM = adj.shape[1]
NUM_ATOMS = adj.shape[-1]
ATOM_DIM = feat.shape[-1]

generator = GraphGenerator(
    BATCH_SIZE, DROPOUT_RATE, LATENT_DIM, [128, 256, 512], BOND_DIM, NUM_ATOMS, ATOM_DIM
).to(device)

discriminator = Discriminator(
    [(128, 128), 128], [128, 1], ATOM_DIM, BOND_DIM, DROPOUT_RATE
).to(device)

optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

adj_tensor, feat_tensor = dataloader.train_batch(adj, feat, BATCH_SIZE)
real_logits = discriminator(adj_tensor[0].to(device), feat_tensor[0].to(device))
print(real_logits)
# def main():
#     indices = torch.arange(0, adj_tensor.shape[0], 1)
#     disc_steps = 5
#     generator.train()
#     discriminator.train()

#     for epoch in tqdm(range(EPOCHS)):
#         gen_running = 0
#         disc_running = 0
#         for idx in indices:
#             graph_real = [adj_tensor[idx], feat_tensor[idx]]

#             for _ in range(disc_steps):
#                 z = torch.normal(mean=0, std=1, size=(BATCH_SIZE, LATENT_DIM))
#                 with torch.no_grad():
#                     graph_fake = generator(z)
#                 real_logits = discriminator(graph_real)
#                 fake_logits = discriminator(graph_fake)
#                 penalty = gradient_penalty(graph_real, graph_fake, BATCH_SIZE, discriminator)
#                 d_loss = disc_loss(real_logits, fake_logits, penalty)

#                 optimizer_disc.zero_grad()
#                 d_loss.backward(retain_graph=True)
#                 optimizer_disc.step()

#                 disc_running += d_loss.item()

#             z = torch.normal(mean=0, std=1, size=(BATCH_SIZE, LATENT_DIM))
#             graph_fake = generator(z)
#             fake_logits = discriminator(graph_fake)
#             g_loss = gen_loss(fake_logits)
#             optimizer_gen.zero_grad()
#             g_loss.backward()
#             optimizer_gen.step()
#             gen_running += g_loss.item()

#         if (epoch % 10 == 0) or (epoch == EPOCHS - 1):
#             save_generator(epoch, generator_path, generator, optimizer_gen, g_loss)
#             save_discriminator(epoch, discriminator_path, discriminator, optimizer_disc, d_loss)
#             image_path = f'model.001/images/epoch.{epoch}'
#             save_images(generator, BATCH_SIZE, LATENT_DIM, image_path, atom_mapping, bond_mapping)

#         avg_gen = gen_running / len(indices)
#         avg_disc = (disc_running / disc_steps) / len(indices)

#         print(f'Discriminator Loss: {avg_disc}, Generator Loss: {avg_gen}')
#         indices = dataloader.shuffle_index(len(indices))
        
# if __name__ == "__main__":
#     main()
