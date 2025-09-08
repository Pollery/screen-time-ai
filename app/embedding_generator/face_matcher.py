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

        for frame_name in tqdm(
            sorted_frame_names, desc="Processing video frames"
        ):
            try:
                img_tensor = self.video_tensors[frame_name]
                # Convert the tensor to a format usable by face_embedder
                img = self.face_embedder.tensor_to_cv2(img_tensor)
                faces = self.face_embedder.detect_faces(img)
                num_faces = len(faces)

                if num_faces == 0:
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

                    all_results.append(
                        {
                            "image_filename": frame_name,
                            "num_faces": num_faces,
                            "best_match": best_match,
                            "similarity_score": best_similarity,
                        }
                    )

            except Exception as e:
                print(f"Error processing frame {frame_name}: {e}")

        df = pd.DataFrame(all_results)
        return df


# You would need to add this method to your FaceEmbeddingGenerator class.
# This is a critical new piece for the module to work as intended.
def add_tensor_to_cv2_method_to_face_embedder():
    from PIL import Image
    import numpy as np

    def tensor_to_cv2(self, img_tensor: torch.Tensor):
        """
        Converts a torch.Tensor (C, H, W) to a BGR NumPy array for OpenCV.
        Assumes normalized tensor input.
        """
        # Denormalize
        mean = (
            torch.tensor([0.485, 0.456, 0.406])
            .view(3, 1, 1)
            .to(img_tensor.device)
        )
        std = (
            torch.tensor([0.229, 0.224, 0.225])
            .view(3, 1, 1)
            .to(img_tensor.device)
        )
        img_tensor = img_tensor * std + mean
        # Permute and convert to uint8
        img_np = (
            img_tensor.permute(1, 2, 0)
            .mul(255)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )
        # Convert RGB to BGR for OpenCV
        img_bgr = np.ascontiguousarray(img_np[:, :, ::-1])
        return img_bgr

    # Attach the method to the class
    # NOTE: In a real scenario, you'd directly add this to the class definition.
    # This is done here for illustrative purposes.
    setattr(
        FaceMatcherModule,
        "tensor_to_cv2",
        tensor_to_cv2,
    )
