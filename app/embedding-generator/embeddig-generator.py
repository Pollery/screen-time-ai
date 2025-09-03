import torch
import os
from pathlib import Path
from fastai.vision.all import *
from tqdm import tqdm

# Path to the directory containing actor images
IMAGES_FOLDER = (
    Path(__file__).resolve().parent.parent / "tmdb-api" / "images_train"
)


def generate_fastai_embeddings(image_folder: str):
    """
    Generates face embeddings for all images in a given folder using a pre-trained FastAI model.

    This function loads a pre-trained ResNet34 model, removes its classification head,
    and uses the remaining 'body' to extract a high-dimensional feature vector (embedding)
    for each face image.

    Args:
        image_folder (str): The path to the folder with face images.

    Returns:
        tuple: A tuple containing:
            - embeddings (torch.Tensor): A tensor of all generated face embeddings.
            - actor_names (list): A list of corresponding actor names.
    """
    embeddings_list = []
    actor_names = []

    try:
        # Create the model's body directly, which is the feature-extraction part.
        # We use 'create_body' to load the pre-trained weights without the final
        # classification layer.
        model = create_body(resnet34, pretrained=True)
        model.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading the pre-trained model: {e}")
        print("Please check your internet connection and try again.")
        return None, None

    # Ensure the images folder exists
    image_path = Path(image_folder)
    if not image_path.exists():
        print(f"Error: The folder '{image_folder}' does not exist.")
        return None, None

    # Get a list of all image files in the folder
    image_files = get_image_files(image_path)
    if not image_files:
        print("No image files found in the specified folder.")
        return None, None

    print(f"Found {len(image_files)} images. Generating embeddings...")

    with torch.no_grad():  # Disable gradient calculation for faster inference
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load and preprocess the image
                img = PILImage.create(img_path)
                # Resize to a standard size for the model
                img_tensor = img.resize((224, 224)).to_tensor().unsqueeze(0)

                # Pass the image through the model to get the embedding
                embedding = model(img_tensor).squeeze()
                embeddings_list.append(embedding)

                # Extract the actor's name from the file name
                actor_name = img_path.stem.replace("_", " ")
                actor_names.append(actor_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Stack all embeddings into a single tensor for efficient storage and use
    if embeddings_list:
        embeddings_tensor = torch.stack(embeddings_list)
        return embeddings_tensor, actor_names
    else:
        return None, None


def save_embeddings(
    embeddings_tensor, actor_names, output_file="fastai_embeddings.pth"
):
    """
    Saves the embeddings and corresponding names to a PyTorch file (.pth).

    This is a highly efficient way to store numerical data, much faster than
    a text-based format like JSON, and it can be easily loaded back into
    a PyTorch environment.

    Args:
        embeddings_tensor (torch.Tensor): The tensor containing all embeddings.
        actor_names (list): The list of actor names.
        output_file (str): The name of the output file.
    """
    torch.save(
        {"embeddings": embeddings_tensor, "names": actor_names}, output_file
    )
    print(f"\nEmbeddings and names saved successfully to '{output_file}'!")


if __name__ == "__main__":
    embeddings, names = generate_fastai_embeddings(IMAGES_FOLDER)
    if embeddings is not None:
        save_embeddings(embeddings, names)
        # You can now load this file later for tasks like face recognition or clustering
        # loaded_data = torch.load('fastai_embeddings.pth')
        # loaded_embeddings = loaded_data['embeddings']
        # loaded_names = loaded_data['names']
        # print(f"\nLoaded {len(loaded_embeddings)} embeddings.")
