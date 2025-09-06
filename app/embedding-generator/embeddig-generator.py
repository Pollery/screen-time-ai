from fastai.vision.all import *
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Define the path to the image directory
path = Path("../tmdb-api/images_train")

# Create DataLoaders using the image folder (used here only for preview purposes)
dls = ImageDataLoaders.from_path_func(
    path,
    get_image_files(path),
    label_func=lambda x: x.stem.replace("_", " "),
)

print("Labels (actors):", dls.vocab)
print("Preview a batch:")
dls.show_batch(max_n=9, figsize=(8, 8))

# Initialize ArcFace model using InsightFace
app_arcface = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app_arcface.prepare(ctx_id=0, det_size=(640, 640))

# Dictionary to store embeddings for each actor label
embeddings_by_actor = {}

# Get all image files
image_files = get_image_files(path)

# Process each image without gradient calculation
with torch.no_grad():
    for img_path in tqdm(image_files, desc="Generating embeddings"):
        try:
            # Load the image using OpenCV
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                print(f"Skipping {img_path}: unable to read the image.")
                continue

            # Create a copy for drawing bounding boxes and landmarks
            img_display = img_cv.copy()

            # Run face analysis (detection & alignment)
            faces = app_arcface.get(img_cv)

            if not faces:
                print(f"No face detected in {img_path}")
                continue

            # Select the face with the largest bounding box area
            largest_face = max(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0])
                * (face.bbox[3] - face.bbox[1]),
            )

            # Get bounding box coordinates
            bbox = largest_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Draw bounding box (Green rectangle)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get landmark points
            kps = largest_face.kps.astype(int)
            for kp in kps:
                cv2.circle(img_display, (kp[0], kp[1]), 2, (0, 0, 255), -1)

            # Get and store the embedding for the largest detected face
            embedding = torch.tensor(largest_face.embedding)
            actor = img_path.stem.replace("_", " ")
            embeddings_by_actor.setdefault(actor, []).append(embedding)

            # Convert BGR image (OpenCV) to RGB for display in Jupyter Notebook
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(f"Detected Face: {img_path.name}")
            plt.show()

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Average embeddings for each actor (if multiple images exist)
actor_embeddings = {
    actor: torch.stack(embs).mean(0)
    for actor, embs in embeddings_by_actor.items()
}

# Save the embeddings to a file
torch.save(actor_embeddings, "actor_embeddings.pth")
print("Saved actor embeddings to actor_embeddings.pth")
