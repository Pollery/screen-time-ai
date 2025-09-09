import torch
import pandas as pd
from dotenv import load_dotenv
import os
import base64
from pathlib import Path
import cv2  # Import for cv2.imencode
import numpy as np  # Import for array handling
from app.frame_extractor.frame_extractor import VideoProcessor
from app.tmdb_api.tmdb_api import TMDbClient, MovieDataProcessor
from app.embedding_generator.face_embedding import FaceEmbeddingGenerator
from app.embedding_generator.face_matcher import FaceMatcherModule
from app.generate_eda_report.generate_eda_report import generate_eda_report

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MOVIE_NAME = "Harry Potter and the Deathly Hallows part 1"
VIDEO_FILE = "data/Harry Potter and the Deathly Hallows - Main Trailer [Su1LOpjvdZ4].webm"
OUTPUT_DIRECTORY = f"{(MOVIE_NAME.replace(" ", "_")
        .replace(":", "")
        .replace("/", "")
        .replace("\\", ""))}"

# Get the API header from environment variables
HEADER = os.getenv("HEADER")


def tensor_to_base64(img_tensor: torch.Tensor, normalized: bool = True) -> str:
    """
    Converts a torch.Tensor image to a base64-encoded JPEG string.
    - normalized=True → assumes ImageNet-style normalization, will denormalize before display.
    - normalized=False → assumes raw RGB [0,1] or [0,255], no denormalization.
    """
    if img_tensor is None:
        return ""
    try:
        tensor = img_tensor.clone()

        if normalized:
            # Denormalize (ImageNet defaults)
            mean = (
                torch.tensor([0.485, 0.456, 0.406])
                .view(3, 1, 1)
                .to(tensor.device)
            )
            std = (
                torch.tensor([0.229, 0.224, 0.225])
                .view(3, 1, 1)
                .to(tensor.device)
            )
            tensor = tensor * std + mean

        # Ensure values are in [0,1] then scale
        tensor = tensor.clamp(0, 1)
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        # OpenCV expects BGR
        ret, buffer = cv2.imencode(".jpg", img_np[:, :, ::-1])
        if not ret:
            print("Could not encode image to JPEG.")
            return ""

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"Error converting tensor to base64: {e}")
        return ""


# --- Step 1: Extract Movie Frames and Tensors ---
print("\n--- Starting Video Processing ---")
try:
    processor = VideoProcessor(VIDEO_FILE, OUTPUT_DIRECTORY)
    video_tensors = processor.process()
except Exception as e:
    print(f"Error during video processing: {e}")
    video_tensors = {}

# --- Step 2: Extract Actor Data from TMDB ---
print("\n--- Starting API Extraction ---")
try:
    client = TMDbClient(api_header=HEADER)
    tmdb_processor = MovieDataProcessor(client)
    actor_tensors, actors_df = tmdb_processor.process_movie(MOVIE_NAME)
except Exception as e:
    print(f"Error during API data extraction: {e}")
    actor_tensors = {}
    actors_df = pd.DataFrame()

# --- Step 3: Generate Face Embeddings for Actors ---
print("\n--- Starting Actors Embedding Process ---")
face_embedder = FaceEmbeddingGenerator()

if actor_tensors:
    print("Processing actor images from TMDb...")
    face_embedder.process_tensors(actor_tensors, display=False)
    actor_embeddings = face_embedder.get_embeddings()
else:
    print("No actor tensors were generated. Skipping actor embedding process.")
    actor_embeddings = {}

# --- Step 4: Perform Face Matching ---
print("\n--- Starting Video Face Matching Process ---")
if video_tensors and actor_embeddings:
    matcher = FaceMatcherModule(
        video_tensors=video_tensors,
        actor_embeddings=actor_embeddings,
        face_embedder=face_embedder,
    )
    face_matching_df = matcher.run()
    print("Face matching complete.")

    # --- Step 5: Merge Results with Actor Info and Base64 Images ---
    if not actors_df.empty:
        face_matching_df["best_match"] = face_matching_df["best_match"].astype(
            actors_df["id"].dtype
        )

        # Add video frame base64 images using the pre-loaded tensors
        # These ARE normalized → need denormalization
        face_matching_df["scene_image_base64"] = (
            face_matching_df["image_filename"]
            .astype(int)
            .apply(
                lambda x: tensor_to_base64(
                    video_tensors.get(x), normalized=True
                )
            )
        )

        # Add actor profile base64 images using the pre-loaded tensors
        # These are NOT normalized → use raw RGB
        actors_df["profile_image_base64"] = (
            actors_df["id"]
            .astype(str)
            .apply(
                lambda x: tensor_to_base64(
                    actor_tensors.get(x), normalized=False
                )
            )
        )

        df_merged = pd.merge(
            face_matching_df,
            actors_df,
            left_on="best_match",
            right_on="id",
            how="left",
        ).drop(columns=["id"])

        output_csv_path = "face_matching_results_with_actor_info.csv"
        df_merged.to_csv(output_csv_path, index=False)
        print(f"Merged results saved to {output_csv_path}")
    else:
        print(
            "Actors DataFrame is empty. Skipping merge and saving raw results."
        )
        output_csv_path = "face_matching_results.csv"
        face_matching_df.to_csv(output_csv_path, index=False)
        print(f"Raw results saved to {output_csv_path}")

else:
    print(
        "Video tensors or actor embeddings are missing. Skipping face matching."
    )


if __name__ == "__main__":
    generate_eda_report(df_merged)
    print("Script execution finished.")
