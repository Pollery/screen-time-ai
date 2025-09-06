import torch
import cv2
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
from face_embedding import FaceEmbeddingGenerator
import base64


def img_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


class FaceMatcher:
    def __init__(
        self, images_dir: Path, embeddings_path: Path, device: str = "cuda:0"
    ):
        """
        images_dir: Directory containing the extracted jpg images.
        embeddings_path: Path to the actor embeddings file (actor_embeddings.pth) where keys are actor ids.
        device: Device to use for face embedding.
        """
        self.images_dir = images_dir
        self.embeddings_path = embeddings_path
        self.device = device
        self.face_embedder = FaceEmbeddingGenerator(device=device)

    def load_actor_embeddings(self):
        # Expecting a dictionary mapping actor id to embedding tensor
        return torch.load(self.embeddings_path)

    def run(self) -> pd.DataFrame:
        actor_embeddings = self.load_actor_embeddings()

        # Getting sorted image paths.
        image_paths = sorted(list(self.images_dir.glob("*.jpg")))

        all_results = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Load image and detect faces
                img = self.face_embedder.load_image(img_path)
                faces = self.face_embedder.detect_faces(img)
                num_faces = len(faces)

                if num_faces == 0:
                    print(f"No face detected in {img_path}")
                    continue

                for face in faces:
                    # Convert face embedding to tensor
                    face_embedding = torch.tensor(face.embedding)
                    best_match = None
                    best_similarity = (
                        -1.0
                    )  # cosine similarity ranges from -1 to 1

                    # Compare this face embedding with each actor embedding using cosine similarity
                    for actor_id, actor_embedding in actor_embeddings.items():
                        sim = F.cosine_similarity(
                            face_embedding.unsqueeze(0),
                            actor_embedding.unsqueeze(0),
                        )
                        sim_value = sim.item()
                        if sim_value > best_similarity:
                            best_similarity = sim_value
                            best_match = actor_id

                    all_results.append(
                        {
                            "image_filename": img_path.name,
                            "num_faces": num_faces,
                            "best_match": best_match,  # now actor id
                            "similarity_score": best_similarity,
                        }
                    )

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        df = pd.DataFrame(all_results)
        print(df)
        return df


def main():
    current_dir = Path(__file__).resolve().parent
    # Path to extracted images
    images_dir = (
        current_dir.parent
        / "frame-extractor"
        / "Harry Potter and the Deathly Hallows - Main Trailer [Su1LOpjvdZ4]"
    )
    # Path to actor embeddings file
    embeddings_path = current_dir / "actor_embeddings.pth"

    face_matcher = FaceMatcher(
        images_dir=images_dir, embeddings_path=embeddings_path, device="cuda:0"
    )
    df = face_matcher.run()

    # Save the face matching results before merging actors info.
    df.to_csv("face_matching_results.csv", index=False)

    # Merge with actors info from actors.csv by matching actor id.
    actor_info_path = (
        current_dir.parent / "tmdb-api" / "actors_info" / "actors.csv"
    )
    actors_df = pd.read_csv(actor_info_path)

    # Ensure the types match before merging (assuming actors_df's id is int64)
    df["best_match"] = df["best_match"].astype(actors_df["id"].dtype)

    df_merged = pd.merge(
        df, actors_df, left_on="best_match", right_on="id", how="left"
    ).drop(columns=["id"])
    df_merged.to_csv("face_matching_results_with_actor_info.csv", index=False)
    print("Merged results saved to face_matching_results_with_actor_info.csv")
    return df_merged


if __name__ == "__main__":
    main()

# The following code is intended for Jupyter Notebook usage.
# It reads the merged CSV, creates an HTML image column and a base64 encoded scene_image column,
# and displays the table with rendered images.
from IPython.display import display, HTML

current_dir = Path(__file__).resolve().parent
images_dir = (
    current_dir.parent
    / "frame-extractor"
    / "Harry Potter and the Deathly Hallows - Main Trailer [Su1LOpjvdZ4]"
)


df = pd.read_csv("face_matching_results_with_actor_info.csv")


def base64_to_img_html(encoded_str):
    if isinstance(encoded_str, str) and encoded_str:
        return (
            f'<img src="data:image/jpeg;base64,{encoded_str}" width="100" />'
        )
    return ""


# Apply the conversion to the 'image' column
df["image"] = df["image"].apply(base64_to_img_html)
# Create a column with base64 encoded image data to display in the notebook
df["scene_image"] = df["image_filename"].apply(
    lambda x: f'<img src="data:image/jpeg;base64,{img_to_base64(images_dir / x)}" width="150">'
)
display(HTML(df.to_html(escape=False)))
