import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


class FaceMatcherModule:
    def __init__(
        self,
        video_tensors: dict,
        actor_embeddings: dict,
        face_embedder: "FaceEmbeddingGenerator",
    ):
        """
        Initializes the FaceMatcherModule with pre-loaded data.

        Args:
            video_tensors: A dictionary mapping frame filenames to their tensor representations.
            actor_embeddings: A dictionary mapping actor IDs to their embedding tensors.
            face_embedder: An instance of the FaceEmbeddingGenerator class.
        """
        self.video_tensors = video_tensors
        self.actor_embeddings = actor_embeddings
        self.face_embedder = face_embedder

    def run(self) -> pd.DataFrame:
        """
        Processes video frames, detects faces, and finds the best matching actor
        for each detected face.

        Returns:
            A pandas DataFrame with the matching results.
        """
        all_results = []
        # Process frames in sorted order for consistent results
        sorted_frame_names = sorted(list(self.video_tensors.keys()))
        with torch.no_grad():
            for frame_name in tqdm(
                sorted_frame_names, desc="Processing video frames in batch"
            ):
                try:
                    img_tensor = self.video_tensors[frame_name]
                    # Convert the tensor to a format usable by face_embedder
                    img = self.face_embedder.tensor_to_cv2(img_tensor)
                    faces = self.face_embedder.detect_faces(img)
                    num_faces = len(faces)

                    if num_faces == 0:
                        del img_tensor
                        del img
                        continue

                    for face in faces:
                        face_embedding = torch.tensor(face.embedding)
                        best_match = None
                        best_similarity = -1.0

                        # Compare with each actor embedding
                        for (
                            actor_id,
                            actor_embedding,
                        ) in self.actor_embeddings.items():
                            sim = F.cosine_similarity(
                                face_embedding.unsqueeze(0),
                                actor_embedding.unsqueeze(0),
                            )
                            sim_value = sim.item()
                            if sim_value > best_similarity:
                                best_similarity = sim_value
                                best_match = actor_id

                        # Clear the face embedding after it's used
                        del face_embedding

                        all_results.append(
                            {
                                "image_filename": frame_name,
                                "num_faces": num_faces,
                                "best_match": best_match,
                                "similarity_score": best_similarity,
                            }
                        )

                    # Clear variables for the entire frame after its faces are processed
                    del img_tensor
                    del img
                    del faces

                except Exception as e:
                    print(f"Error processing frame {frame_name}: {e}")

        df = pd.DataFrame(all_results)
        return df
