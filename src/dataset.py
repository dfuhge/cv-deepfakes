
import random
import torch
import numpy as np
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset, DataLoader


class FrameDataset(Dataset):
    def __init__(
        self,
        frames: list[Image.Image],
        image_size: int = 128
    ):
        self.frames = frames
        self.image_size = image_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]

        if not isinstance(frame, Image.Image):
            raise TypeError("Expected a PIL Image.")

        # Resize and convert to RGB
        frame = frame.convert("RGB").resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # Convert to tensor, shape (3, H, W)
        array = np.array(frame, dtype=np.uint8, copy=True)

        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .float() / 255.0
        )

        return tensor
