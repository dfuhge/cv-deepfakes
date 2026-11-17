# Pipeline for Data Processing

import math
import matplotlib.pyplot as plt
import subprocess
import tempfile
import re
import numpy as np
import torch
import yt_dlp
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from yt_dlp.utils import download_range_func
from PIL import Image


def read_input_parameters() -> tuple:

    while True:
        VIDEO_URL = input("Video URL: ")
        TIMESTAMP = int(input("Starting from (in s): "))
        DURATION = int(input("Duration (in s): "))

        results = (VIDEO_URL, TIMESTAMP, DURATION)
        if check_validity(results[0], results[1], results[2]):
            print("Inputs valid")
            return results
            break

def check_validity(url: str, time: str, dur: int) -> bool:
    print(f"Checking validity for URL={url} at time {time} and duration {dur}")

    # Validate general time and duration
    if not all(math.isfinite(x) for x in (time, dur)):
        print("Infinite numbers")
        return False
    if time < 0 or dur < 0:
        print("Negative numbers")
        return False

    # Validate URL
    parsed = urlparse(url)
    youtube_hosts = {
        "youtube.com", "www.youtube.com",
        "m.youtube.com", "music.youtube.com",
        "youtu.be", "www.youtu.be",
        "youtube-nocookie.com", "www.youtube-nocookie.com",
    }
    if parsed.scheme not in ("http", "https"):
        print("Unknown scheme")
        return False
    if parsed.hostname not in youtube_hosts:
        print("Unknown host")
        return False

    # Extract video ID
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        parts = parsed.path.strip("/").split("/")
        video_id = parts[0] if len(parts) == 1 else None
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    else:
        parts = parsed.path.strip("/").split("/")
        video_id = (
            parts[1]
            if len(parts) == 2
            and parts[0] in ("shorts", "live", "embed", "v")
            else None
        )

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        print("Unknown video id")
        return False

    # 3. Retrieve actual video duration
    try:
        with yt_dlp.YoutubeDL({
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
        }) as ydl:
            info = ydl.extract_info(url, download=False)

        video_duration = info.get("duration") if info else None

        if video_duration is None or not math.isfinite(video_duration):
            print("Unknown duration")
            return False

        # 4. Check timestamps
        return time < video_duration and time + dur <= video_duration

    except yt_dlp.utils.DownloadError:
        print("Download error")
        return False

def sample_youtube_frames(url: str, timestamp: float, duration: float, sample_interval: float = 1.0) -> list[Image.Image]:
    """
    Sample images from a YouTube video segment.

    Args:
        url: YouTube video URL.
        timestamp: Start time in seconds.
        duration: Segment duration in seconds.
        sample_interval: Seconds between sampled frames.

    Returns:
        List of PIL Images.
    """

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 2. Download only the requested segment
        # For zero duration, extract a single image.
        end_time = timestamp + (duration if duration > 0 else 1.0)

        options = {
            "format": "bestvideo[ext=mp4]/bestvideo/best",
            "outtmpl": str(tmp_path / "clip.%(ext)s"),
            "download_ranges": download_range_func(
                None, [(timestamp, end_time)]
            ),
            "force_keyframes_at_cuts": True,
            "noplaylist": True,
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])

        # Locate downloaded video
        video_files = [
            p for p in tmp_path.iterdir()
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".ts"}
        ]

        if not video_files:
            raise RuntimeError("Video segment could not be downloaded.")

        video_path = video_files[0]

        # 3. Extract frames
        frame_pattern = str(tmp_path / "frame_%05d.jpg")

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
        ]

        if duration > 0:
            command += [
                "-vf", f"fps={1 / sample_interval}:start_time=0",
                "-frames:v", str(math.ceil(duration / sample_interval)),
            ]
        else:
            command += ["-frames:v", "1"]

        command += ["-q:v", "2", frame_pattern]

        subprocess.run(command, check=True)

        # 4. Load frames into memory
        frames = []

        for frame_file in sorted(tmp_path.glob("frame_*.jpg")):
            with Image.open(frame_file) as img:
                frames.append(img.convert("RGB"))

        return frames

def visualize_frames(frames: list[Image.Image], n: int = 5, cols: int = 5):
    """Visualize the first n frames in a grid."""

    n = min(n, len(frames))
    if n <= 0:
        return

    cols = min(cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = [axes] if n == 1 else axes.flatten()

    for i in range(n):
        axes[i].imshow(frames[i])
        axes[i].set_title(f"Frame {i + 1}")
        axes[i].axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def frame_to_tensor(
    frame: Image.Image,
    size: int = 128
) -> torch.Tensor:
    """Convert PIL image into a normalized PyTorch batch."""

    frame = frame.convert("RGB").resize((size, size))

    array = np.asarray(frame).copy()

    tensor = torch.from_numpy(array).float() / 255.0

    return tensor.permute(2, 0, 1).unsqueeze(0)


def tensor_to_frame(tensor: torch.Tensor) -> Image.Image:
    """Convert model output back into a PIL image."""

    array = (
        tensor.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .numpy()
    )

    array = (array * 255).astype(np.uint8)

    return Image.fromarray(array)

if __name__ == "__main__":
    print("MAIN")
    video = read_input_parameters()
    frames = sample_youtube_frames(video[0], video[1], video[2])
    visualize_frames(frames)
