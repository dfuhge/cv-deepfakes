import cv2
import math
import numpy as np
import shutil
import subprocess
import tempfile
import torch
import yt_dlp
from pathlib import Path
import re
from training import train_autoencoders, save_model, load_model
from data_processing import sample_youtube_frames, check_validity


def train_from_youtube(
    video_urls: dict[str, str],
    model_dir: str = "models",
    timestamp: float = 0.0,
    duration: float = 30.0,
    sample_interval: float = 0.5,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 128
) -> tuple[dict, dict, dict]:
    """
    Train autoencoders from YouTube videos.

    Each video gets an individual decoder.
    All decoders share one encoder.

    Returns:
        models: Trained autoencoder dictionary.
        history: Loss history per video.
        model_paths: Saved checkpoint paths.
    """

    if not video_urls:
        raise ValueError("video_urls must not be empty.")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and sample frames
    images = {}

    for name, url in video_urls.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid model name: {name}")

        print(f"Extracting frames from {name}...")

        frames = sample_youtube_frames(
            url=url,
            timestamp=timestamp,
            duration=duration,
            sample_interval=sample_interval
        )

        if not frames:
            raise ValueError(f"No frames extracted from {name}")

        images[name] = frames
        print(f"Extracted {len(frames)} frames")

    # 2. Train shared encoder and individual decoders
    models, history = train_autoencoders(
        images=images,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size
    )

    # 3. Save trained models
    model_paths = {}

    for name, model in models.items():
        path = model_dir / f"{name}.pt"

        save_model(model, str(path))
        model_paths[name] = str(path)

        print(f"Saved {name} to {path}")

    return models, history, model_paths





def inference_youtube_video(
    url: str,
    model_path: str,
    output_path: str = "output.mp4",
    image_size: int = 128,
    batch_size: int = 16,
    preserve_audio: bool = True
) -> str:
    """
    Run a trained autoencoder on every frame of a YouTube video.

    Args:
        url: Source YouTube video URL.
        model_path: Path to the saved autoencoder.
        output_path: Path for the generated MP4.
        image_size: Model input resolution.
        batch_size: Number of frames per inference batch.
        preserve_audio: Copy original audio into the output.

    Returns:
        Path to the generated video.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if image_size < 16 or image_size % 16 != 0:
        raise ValueError("image_size must be divisible by 16.")

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output file must have .mp4 extension.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Validate YouTube URL
    if not check_validity(url, 0, 0):
        raise ValueError("Invalid or unavailable YouTube video.")

    # 2. Load trained autoencoder
    model = load_model(model_path)
    model.eval()

    device = next(model.parameters()).device

    print(f"Running inference on {device}")

    # 3. Download source video
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        options = {
            "format": (
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
                "/best[ext=mp4]/best"
            ),
            "merge_output_format": "mp4",
            "outtmpl": str(tmp_dir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])

        video_files = [
            p for p in tmp_dir.glob("source.*")
            if p.suffix.lower() in {".mp4", ".webm", ".mkv"}
        ]

        if not video_files:
            raise RuntimeError("Video download failed.")

        video_files.sort(key=lambda p: p.suffix != ".mp4")
        source_video = video_files[0]

        # 4. Open video and retrieve metadata
        capture = cv2.VideoCapture(str(source_video))

        if not capture.isOpened():
            raise RuntimeError("Could not open downloaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError("Could not determine video FPS.")

        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Invalid video dimensions.")

        # 5. Create temporary output video without audio
        silent_video = tmp_dir / "processed.mp4"

        writer = cv2.VideoWriter(
            str(silent_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create output video.")

        # 6. Process frames in batches
        @torch.inference_mode()
        def process_batch(frames: list[np.ndarray]) -> None:

            # Resize and convert BGR to RGB
            resized = [
                cv2.cvtColor(
                    cv2.resize(
                        frame,
                        (image_size, image_size),
                        interpolation=cv2.INTER_AREA
                    ),
                    cv2.COLOR_BGR2RGB
                )
                for frame in frames
            ]

            # Convert to tensor: (B, 3, H, W)
            array = np.stack(resized)

            x = (
                torch.from_numpy(array)
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )

            # Encoder -> Decoder
            output = model(x)

            # Convert back to RGB images
            output = (
                output.clamp(0, 1)
                .mul(255)
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )

            # Restore original resolution and write frames
            for image in output:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )

                image = cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR
                )

                writer.write(image)

        print("Processing video frames...")

        frame_buffer = []
        processed_frames = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                frame_buffer.append(frame)

                if len(frame_buffer) >= batch_size:
                    process_batch(frame_buffer)
                    processed_frames += len(frame_buffer)
                    frame_buffer.clear()

            # Process remaining frames
            if frame_buffer:
                process_batch(frame_buffer)
                processed_frames += len(frame_buffer)

        finally:
            capture.release()
            writer.release()

        if processed_frames == 0:
            raise RuntimeError("No video frames could be processed.")

        # 7. Add original audio
        if preserve_audio:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-i", str(silent_video),
                    "-i", str(source_video),
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(output_path)
                ],
                check=True
            )
        else:
            shutil.copyfile(silent_video, output_path)

    print(f"Processed {processed_frames} frames")
    print(f"Saved video to {output_path}")

    return str(output_path)



if __name__ == "__main__":
    video_urls = {
        "person_a": "https://www.youtube.com/watch?v=6v_LVsOl-uo",
        "person_b": "https://www.youtube.com/watch?v=LQtV-iOwijY"
    }

    models, history, model_paths = train_from_youtube(
        video_urls=video_urls,
        model_dir="models",
        timestamp=10,
        duration=120,
        sample_interval=0.5,
        epochs=50,
        batch_size=16
    )