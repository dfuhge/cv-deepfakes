import torch
import random
import matplotlib.pyplot as plt
from models import FaceDecoder, FaceEncoder, FaceAutoencoder
from dataset import FrameDataset
from PIL import Image
from torch.utils.data import DataLoader
from itertools import chain
import torch.nn as nn
import numpy as np

device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def initialize_autoencoders(images: dict):
    

    # Initialize shared encoder
    encoder = FaceEncoder().to(device)

    # Initialize individual decoders and models
    models = {}
    for k in images.keys():
        decoder = FaceDecoder().to(device)
        model = FaceAutoencoder(encoder, decoder)
        models[k] = model
    return models

def initialize_optimizer_loss(models: dict):
    encoder = next(iter(models.values())).encoder
    if any(m.encoder is not encoder for m in models.values()):
        raise ValueError("Models must share the same encoder.")
    optimizer = torch.optim.Adam(chain(
        encoder.parameters(), *(m.decoder.parameters() for m in models.values())
    ), lr=1e-4)
    criterion = nn.L1Loss()
    return optimizer, criterion
        

def train_step(data: torch.Tensor, encoder: FaceEncoder, decoder: FaceDecoder, optimizer: torch.optim.Adam, criterion = nn.L1Loss()):
    encoder.train()
    decoder.train()
    data = data.to(device)
    optimizer.zero_grad()

    latent = encoder(data)
    reconstructed = decoder(latent)

    loss = criterion(reconstructed, data)

    loss.backward()
    optimizer.step()

    return loss.item()

def plot_training_history(history: dict[str, list[float]]):
    plt.figure(figsize=(10, 5))

    for name, losses in history.items():
        plt.plot(
            range(1, len(losses) + 1),
            losses,
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (L1)")
    plt.title("Autoencoder Training")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_autoencoders(
    images: dict[str, list[Image.Image]],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 128,
    device: torch.device | None = None
) -> tuple[dict[str, FaceAutoencoder], dict[str, list[float]]]:
    """
    Train a shared encoder with individual decoders.

    Args:
        images: Dictionary mapping names to PIL image lists.
        epochs: Number of training epochs.
        batch_size: Number of images per training batch.
        lr: Learning rate.
        image_size: Input image resolution.
        device: Training device (optional).

    Returns:
        models: Dictionary of trained autoencoders.
        history: Dictionary of loss histories.
    """

    # 1. Validate arguments
    if not images or any(len(v) == 0 for v in images.values()):
        raise ValueError("All image lists must be non-empty.")

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    if image_size < 16 or image_size % 16 != 0:
        raise ValueError("image_size must be divisible by 16.")

    print(f"Training on {device}")

    # 3. Initialize models
    models = initialize_autoencoders(images)

    # 4. Initialize optimizer and loss
    optimizer, criterion = initialize_optimizer_loss(models, lr)

    # 5. Create DataLoaders
    loaders = {
        name: DataLoader(
            FrameDataset(frames, image_size),
            batch_size=batch_size,
            shuffle=True
        )
        for name, frames in images.items()
    }

    # 6. Initialize loss histories
    history = {name: [] for name in images}

    # 7. Training loop
    for epoch in range(epochs):

        # Create an iterator for each image list
        iterators = {
            name: iter(loader)
            for name, loader in loaders.items()
        }

        # Include every batch from every dataset
        batch_order = [
            name
            for name, loader in loaders.items()
            for _ in range(len(loader))
        ]

        # Mix training across decoders
        random.shuffle(batch_order)

        loss_sums = {name: 0.0 for name in images}
        counts = {name: 0 for name in images}

        for name in batch_order:

            # Get the next batch for this model
            batch = next(iterators[name]).to(device)

            # Update shared encoder + selected decoder
            loss = train_step(
                data=batch,
                model=models[name],
                optimizer=optimizer,
                criterion=criterion
            )

            # Accumulate sample-weighted loss
            loss_sums[name] += loss * batch.size(0)
            counts[name] += batch.size(0)

        # Compute average loss for each decoder
        for name in images:
            avg_loss = loss_sums[name] / counts[name]
            history[name].append(avg_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | " +
            " | ".join(
                f"{name}: {history[name][-1]:.4f}"
                for name in images
            )
        )
    plot_training_history(history)
    return models, history


def reconstruct_frame(
    frame: Image.Image,
    model: FaceAutoencoder,
    image_size: int = 128
) -> Image.Image:

    model.eval()
    model_device = next(model.parameters()).device

    dataset = FrameDataset([frame], image_size)
    x = dataset[0].unsqueeze(0).to(model_device)

    with torch.no_grad():
        output = model(x)

    output = (
        output.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    output = (np.clip(output, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(output)
