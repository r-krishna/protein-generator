import os
from datetime import datetime

import torch
import wandb
from torch.optim import RMSprop
from torch.data import DataLoader

from se3_transformer.model.transformer import SE3Transformer, SE3TransformerPooled
from se3_transformer.model.fiber import Fiber

from protein_generator.datasets import ProteinChainDataset, RandomNoiseDataset
from protein_generator.losses.base import wasserstein_critic_loss, wasserstein_generator_loss


def critic_train_loop(generator, critic, real_data_loader, noise_data_loader, C_optimizer):
    """
    Training loop for the critic
    """
    generator.eval()
    running_loss = 0
    for real_examples, noise in zip(real_data_loader, noise_data_loader):
        generated_examples = generator(*noise)
        C_optimizer.zero_grad()
        loss = wasserstein_critic_loss(generated_examples, real_examples, critic)
        loss.backwards()
        C_optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / (len(real_data_loader) + len(noise_data_loader))
    return avg_loss

def generator_train_loop(generator, critic, noise_data_loader, G_optimizer):
    """
    Training loop for the generator
    """
    critic.eval()
    running_loss = 0
    for noise in noise_data_loader:
        generated_examples = generator(*noise)
        G_optimizer.zero_grad()
        loss = wasserstein_generator_loss(generated_examples, critic)
        loss.backwards()
        G_optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss/(len(noise_data_loader))
    return avg_loss


if __name__ == "main":
    wandb.init(project="protein-generator", entity="rkrishna3")

    # Set up datasets
    pdb_dir = None
    real_data = ProteinChainDataset("data/pdbbind_list.txt", pdb_dir)
    example = real_data[0]
    num_node_feats, num_edge_feats = example[1]["0"].shape[1], example[2]["0"].shape[1]
    noise_data = RandomNoiseDataset(5000, 100, 400, num_node_feats, num_edge_feats)

    real_data_loader = DataLoader(real_data, batch_size=4, shuffle=True)
    noise_data_loader = DataLoader(noise_data, batch_size=4, shuffle=False)

    # Set up models
    num_degrees = 3
    num_channels = 64
    generator = SE3Transformer(
        fiber_in=Fiber({0: num_node_feats}),
        fiber_out=Fiber({0: num_node_feats}),
        fiber_edge=Fiber({0: num_edge_feats}),
    )
    critic = SE3TransformerPooled(
        fiber_in=Fiber({0: num_node_feats}),
        fiber_out=Fiber({0: num_degrees * num_channels}),
        fiber_edge=Fiber({0: num_edge_feats}),
        output_dim=1,
    )

    # Set up gradient descent
    G_optimizer = RMSprop(generator.parameters(), lr=5e-5)
    C_optimizer = RMSprop(critic.parameters(), lr=5e-5)

    # Set up checkpoint saving path
    date = datetime.now().strftime("%Y_%m_%d")
    checkpoint_path = f"trained_models/{date}"

    # Log hyperparameters on wandb

    wandb.config = {
        "G_learning_rate": 5e-5,
        "C_learning_rate": 5e-5,
        "epochs": 100,
        "batch_size": 4
    }

    # training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        wandb.watch(critic)
        wandb.watch(generator)
        # train the discriminator more that the generator
        for _ in range(5):
            critic_loss = critic_train_loop(generator, critic, real_data_loader, noise_data_loader, C_optimizer)
        
        wandb.log({"critic_loss": critic_loss})

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": critic.state_dict(),
                "optimizer_state_dict": C_optimizer.state_dict(),
                "loss": critic_loss 
            }, os.path.join(checkpoint_path, f"critic_{epoch}.pt")
        )
        
        # train the generator
        generator_loss = generator_train_loop(generator, critic, noise_data_loader, G_optimizer)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": generator.state_dict(),
                "optimizer_state_dict": G_optimizer.state_dict(),
                "loss": generator_loss 
            }, os.path.join(checkpoint_path, f"generator_{epoch}.pt")
        )
        
        wandb.log({"generator_loss": generator_loss})
