from fastai.vision.all import *
from pathlib import Path
from face_embedding import FaceEmbeddingGenerator

# Define the path to the image directory
path = Path("../tmdb-api/images_train")

# Create DataLoaders (used here only for preview purposes)
dls = ImageDataLoaders.from_path_func(
    path,
    get_image_files(path),
    label_func=lambda x: x.stem.replace("_", " "),
)

print("Labels (actors):", dls.vocab)
print("Preview a batch:")
dls.show_batch(max_n=9, figsize=(8, 8))

# Initialize the FaceEmbeddingGenerator instance
face_embedder = FaceEmbeddingGenerator()

# Process images in the directory (display enabled for visual feedback)
face_embedder.process_directory(path, display=False)

# Save the embeddings to a file
face_embedder.save_embeddings("actor_embeddings.pth")
