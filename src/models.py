import torch
import torch.nn as nn



class FaceEncoder(nn.Module):
    def __init__(self, latent_channels: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FaceDecoder(nn.Module):
    def __init__(self, latent_channels: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 256, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class FaceAutoencoder(nn.Module):
    def __init__(self, encoder: FaceEncoder, decoder: FaceDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

