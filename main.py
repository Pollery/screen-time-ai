import argparse
import pandas as pd
from dotenv import load_dotenv
import os
import base64
from pathlib import Path
import cv2
import torch
import numpy as np

# Import your existing modules
from app.frame_extractor.frame_extractor import VideoProcessor
from app.tmdb_api.tmdb_api import TMDbClient, MovieDataProcessor
from app.embedding_generator.face_embedding import FaceEmbeddingGenerator
from app.embedding_generator.face_matcher import FaceMatcherModule
from app.generate_eda_report.generate_eda_report import generate_eda_report


# Load environment variables once
load_dotenv()
HEADER = os.getenv("HEADER")


def tensor_to_base64(img_tensor: torch.Tensor, normalized: bool = True) -> str:
    """
    Converts a torch.Tensor image to a base64-encoded JPEG string.
    This function is used for embedding images within the report.
    """
    if img_tensor is None:
        return ""
    try:
        tensor = img_tensor.clone()

        if normalized:
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

        tensor = tensor.clamp(0, 1)
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        ret, buffer = cv2.imencode(".jpg", img_np[:, :, ::-1])
        if not ret:
            print("Could not encode image to JPEG.")
            return ""

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"Error converting tensor to base64: {e}")
        return ""


movie_name = "Top Gun Maverick"
video_path = "./data/Top Gun Maverick (2022) [1080p] [BluRay] [5.1] [YTS.MX]/reduzido15min.mp4"

output_directory = f"{(movie_name.replace(" ", "_").replace(":", "").replace("/", "").replace("\\", ""))}"

# --- Step 1: Extract Movie Frames and Tensors ---
print("🎬 Processing video and extracting frames...")
try:
    processor = VideoProcessor(video_path, output_directory)
    print("✅ Video processing initialized.")
except Exception as e:
    print(f"❌ Error during video processing: {e}")


# --- Step 2: Extract Actor Data from TMDB ---
print("📡 Fetching actor data from TMDB...")
try:
    client = TMDbClient(api_header=HEADER)
    tmdb_processor = MovieDataProcessor(client)
    actor_tensors, actors_df = tmdb_processor.process_movie(movie_name)
    print("✅ API data extraction complete!")
except Exception as e:
    print(f"❌ Error during API data extraction: {e}")

# --- Step 3 & 4: Generate Embeddings and Match Faces (Batch Processing) ---
if not actor_tensors:
    print("⚠️ Skipping face matching due to missing actor data.")


face_embedder = FaceEmbeddingGenerator()
print("🔍 Generating face embeddings and matching faces...")
try:
    face_embedder.process_tensors(actor_tensors, display=False)
    actor_embeddings = face_embedder.get_embeddings()

    face_matching_df = pd.DataFrame()  # Initialize an empty DataFrame
    # Iterate over batches of video tensors
    for i, video_tensors_batch in enumerate(processor.process(batch_size=500)):
        print(f"Processing batch {i+1}...")
        if not video_tensors_batch:
            continue

        matcher = FaceMatcherModule(
            video_tensors=video_tensors_batch,
            actor_embeddings=actor_embeddings,
            face_embedder=face_embedder,
        )
        batch_face_matching_df = matcher.run()
        face_matching_df = pd.concat(
            [face_matching_df, batch_face_matching_df], ignore_index=True
        )

    if face_matching_df.empty:
        print("⚠️ No face matching results generated.")
    print("✅ Face matching complete!")
except Exception as e:
    print(f"❌ Error during face matching: {e}")


# --- Step 5: Add Base64 Images and Merge Results ---
print("💾 Merging results and saving data...")
try:
    if not actors_df.empty:
        # Add actor profile base64 images only if actor data exists
        actors_df["profile_image_base64"] = (
            actors_df["id"]
            .astype(str)
            .apply(
                lambda x: tensor_to_base64(
                    actor_tensors.get(x), normalized=False
                )
            )
        )

        # Ensure data types match before merging
        face_matching_df["best_match"] = face_matching_df["best_match"].astype(
            actors_df["id"].dtype
        )

        df_merged = pd.merge(
            face_matching_df,
            actors_df,
            left_on="best_match",
            right_on="id",
            how="left",
        ).drop(columns=["id"])

        output_csv_path = "face_matching_results_with_actor_info.csv"
        print(f"Merged results saved to {output_csv_path}")
    else:
        print("⚠️ Actors DataFrame is empty. Skipping merge with actor info.")
        df_merged = face_matching_df
        output_csv_path = "face_matching_results.csv"

    df_merged.to_csv(output_csv_path, index=False)
except Exception as e:
    print(f"❌ Error during data merging or saving: {e}")


# --- Step 6: Generate EDA Report ---
print("📊 Generating EDA report...")
try:
    eda_report_path = f"eda_report_{movie_name.replace(" ", "_").replace(":", "").replace("/", "").replace("\\", "").lower()}.md"
    generate_eda_report(df_merged, output_path=eda_report_path)
    print("🎉 EDA report generated successfully!")

except Exception as e:
    print(f"❌ Error during EDA report generation: {e}")
