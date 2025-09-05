from fastai.vision.all import *
import torch
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

# Define the path to the image directory
path = Path("../tmdb-api/images_train")


# Create DataLoaders using the image folder
dls = ImageDataLoaders.from_path_func(
    path,
    get_image_files(path),
    label_func=lambda x: x.stem.replace("_", " "),
    item_tfms=Resize(160),
)


print("Labels (actors):", dls.vocab)
print("Preview a batch:")
dls.show_batch(max_n=9, figsize=(8, 8))

# Load the pretrained Facenet model
model = InceptionResnetV1(pretrained="vggface2").eval()

# Dictionary to store embeddings for each actor label
embeddings_by_actor = {}

# Get all image files
image_files = get_image_files(path)

# Disable gradient calculation for faster inference
with torch.no_grad():
    for img_path in tqdm(image_files, desc="Generating embeddings"):
        try:
            # Load image using fastai's PILImage
            img = PILImage.create(img_path)
            # Resize the image to 160x160
            img = img.resize((160, 160))
            # Convert the PIL image to a tensor using torchvision
            img_tensor = TF.to_tensor(img).unsqueeze(0)
            # Normalize pixel values from [0,1] to [-1,1] as required by Facenet
            img_tensor = (img_tensor - 0.5) / 0.5
            # Get the embedding from the model
            embedding = model(img_tensor).squeeze()
            # Extract actor label from filename
            actor = img_path.stem.replace("_", " ")
            embeddings_by_actor.setdefault(actor, []).append(embedding)
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
