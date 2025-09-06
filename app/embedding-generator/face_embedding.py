from fastai.vision.all import get_image_files
import torch
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from pathlib import Path
from tqdm import tqdm


class FaceEmbeddingGenerator:
    def __init__(self, device="cuda:0"):
        self.app_arcface = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app_arcface.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.embeddings_by_label = {}

    def load_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Unable to read image: {img_path}")
        return img

    def detect_faces(self, img):
        faces = self.app_arcface.get(img)
        return faces

    def draw_faces(self, img, faces):
        img_display = img.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            kps = face.kps.astype(int)
            for kp in kps:
                cv2.circle(img_display, (kp[0], kp[1]), 2, (0, 0, 255), -1)
        return img_display

    def get_embeddings_from_faces(self, faces):
        embeddings = []
        for face in faces:
            emb = torch.tensor(face.embedding)
            embeddings.append(emb)
        return embeddings

    def process_image(self, img_path, display=True):
        """
        Process a single image:
         - Load image
         - Detect faces and select only the largest face
         - Draw bounding box and landmarks on the selected face
         - Get embedding for the selected face
         - Use the image file's stem (with "_" replaced by " ") as the label
         - Optionally display the image with drawn box
        Returns the embedding for the selected face.
        """
        img = self.load_image(img_path)
        faces = self.detect_faces(img)
        if not faces:
            print(f"No face detected in {img_path}")
            return []
        # Select the largest face based on bounding box area
        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1]),
        )
        drawn_img = self.draw_faces(img, [largest_face])
        if display:
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(f"Detected Largest Face: {img_path.name}")
            plt.show()
        # Get embedding for the largest face only
        embedding = torch.tensor(largest_face.embedding)
        label = Path(img_path).stem.replace("_", " ")
        self.embeddings_by_label.setdefault(label, []).append(embedding)
        return [embedding]

    def process_directory(self, directory, display=True):
        """
        Process all images in a directory.
        """
        directory = Path(directory)
        image_files = get_image_files(directory)
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                self.process_image(img_path, display=display)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    def average_embeddings(self):
        """
        Average embeddings for labels with multiple embeddings.
        """
        return {
            label: torch.stack(embs).mean(0)
            for label, embs in self.embeddings_by_label.items()
            if embs
        }

    def save_embeddings(self, filename="actor_embeddings.pth"):
        """
        Save averaged embeddings to a file.
        """
        averaged = self.average_embeddings()
        torch.save(averaged, filename)
        print(f"Saved embeddings to {filename}")
