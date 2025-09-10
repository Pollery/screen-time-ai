# app.py
import streamlit as st
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
    ...
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


# Use st.cache_data to cache the heavy processing functions if needed, though for a full run it's less critical.
# Let's refactor the main logic into a single function.
def run_full_pipeline(movie_name: str, video_path: str):
    output_directory = f"{(movie_name.replace(" ", "_").replace(":", "").replace("/", "").replace("\\", ""))}"

    # --- Step 1: Extract Movie Frames and Tensors ---
    with st.status(
        "🎬 Processing video and extracting frames...", expanded=True
    ) as status:
        try:
            processor = VideoProcessor(video_path, output_directory)
            # video_tensors = processor.process() # Original line, now handled by iteration
            # status.update(
            #     label="✅ Video processing complete!", state="complete"
            # )
        except Exception as e:
            st.error(f"Error during video processing: {e}")
            status.update(label="❌ Video processing failed.", state="error")
            return None

    # --- Step 2: Extract Actor Data from TMDB ---
    with st.status(
        "📡 Fetching actor data from TMDB...", expanded=True
    ) as status:
        try:
            client = TMDbClient(api_header=HEADER)
            tmdb_processor = MovieDataProcessor(client)
            actor_tensors, actors_df = tmdb_processor.process_movie(movie_name)
            status.update(
                label="✅ API data extraction complete!", state="complete"
            )
        except Exception as e:
            st.error(f"Error during API data extraction: {e}")
            status.update(
                label="❌ API data extraction failed.", state="error"
            )
            return None

    # --- Step 3 & 4: Generate Embeddings and Match Faces (Batch Processing) ---
    if not actor_tensors:
        st.warning("Skipping face matching due to missing actor data.")
        return None

    face_embedder = FaceEmbeddingGenerator()
    with st.status(
        "🔍 Generating face embeddings and matching faces...", expanded=True
    ) as status:
        try:
            face_embedder.process_tensors(actor_tensors, display=False)
            actor_embeddings = face_embedder.get_embeddings()

            if not actor_embeddings:
                st.warning(
                    "No actor embeddings generated. Skipping face matching."
                )
                return None

            all_face_matching_results = []
            # Iterate over batches of video tensors
            for i, video_tensors_batch in enumerate(
                processor.process(batch_size=100)
            ):
                st.write(f"Processing batch {i+1}...")
                if not video_tensors_batch:
                    continue

                matcher = FaceMatcherModule(
                    video_tensors=video_tensors_batch,
                    actor_embeddings=actor_embeddings,
                    face_embedder=face_embedder,
                )
                batch_face_matching_df = matcher.run()
                all_face_matching_results.append(batch_face_matching_df)

            if not all_face_matching_results:
                st.warning("No face matching results generated.")
                return None

            face_matching_df = pd.concat(
                all_face_matching_results, ignore_index=True
            )
            status.update(label="✅ Face matching complete!", state="complete")
        except Exception as e:
            st.error(f"Error during face matching: {e}")
            status.update(label="❌ Face matching failed.", state="error")
            return None

        # --- Step 5: Add Base64 Images and Merge Results ---
        with st.spinner("💾 Merging results and saving data..."):

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

                face_matching_df["best_match"] = face_matching_df[
                    "best_match"
                ].astype(actors_df["id"].dtype)

                df_merged = pd.merge(
                    face_matching_df,
                    actors_df,
                    left_on="best_match",
                    right_on="id",
                    how="left",
                ).drop(columns=["id"])

                output_csv_path = "face_matching_results_with_actor_info.csv"
                st.success(f"Merged results saved to {output_csv_path}")
            else:
                st.warning(
                    "Actors DataFrame is empty. Skipping merge with actor info."
                )
                df_merged = face_matching_df
                output_csv_path = "face_matching_results.csv"

            df_merged.to_csv(
                output_csv_path, index=False
            )  # Save the final DataFrame

        # --- Step 6: Generate EDA Report ---
        with st.spinner("📊 Generating EDA report..."):
            eda_report_path = (
                f"eda_report_{movie_name.replace(' ', '_').lower()}.md"
            )
            generate_eda_report(df_merged, output_path=eda_report_path)
            st.success("🎉 EDA report generated successfully!")
            return eda_report_path


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Movie Face Recognition EDA", page_icon="🎬", layout="wide"
)

st.title("🎬 Movie Face Recognition & EDA App")

# Create a state variable to track if the report is ready
if "report_path" not in st.session_state:
    st.session_state.report_path = None
    st.session_state.show_results = False

# Sidebar for user inputs
with st.sidebar:
    st.header("Upload Video & Enter Movie Name")
    movie_name = st.text_input(
        "Enter Movie Name (e.g., Harry Potter and the Deathly Hallows part 1)",
        "Harry Potter and the Deathly Hallows part 1",
    )
    uploaded_file = st.file_uploader(
        "Upload a Movie Trailer or Clip", type=["mp4", "webm", "mov"]
    )
    st.info(
        "The application will process the video, identify actors, and generate an EDA report."
    )

if "report_path" not in st.session_state:
    st.session_state.report_path = None
    st.session_state.show_results = False

if st.button("🚀 Start Analysis"):
    if uploaded_file is not None and movie_name:
        # Save the uploaded file temporarily
        video_path = Path("temp_uploaded_video.webm")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.report_path = None  # Reset previous report
        st.session_state.show_results = False

        with st.container():
            st.subheader("Process Status")
            report_path = run_full_pipeline(movie_name, str(video_path))
            if report_path:
                st.session_state.report_path = report_path
                st.session_state.show_results = True

            # Clean up temporary file
            os.remove(video_path)
    else:
        st.error("Please provide both a movie name and a video file.")

# Display the EDA report if it exists
if st.session_state.show_results and st.session_state.report_path:
    st.markdown("---")
    st.header("📊 EDA Report")
    try:
        with open(st.session_state.report_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        st.markdown(markdown_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Report not found. Please run the analysis again.")
if st.session_state.show_results and st.session_state.report_path:
    st.markdown("---")
    st.header("📊 EDA Report")
    try:
        with open(st.session_state.report_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        st.markdown(markdown_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Report not found. Please run the analysis again.")
