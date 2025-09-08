from fastai.vision.all import get_image_files
import torch
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import numpy as np


class FaceEmbeddingGenerator:
    def __init__(self, device="cuda:0"):
        ctx_id = 0 if device.startswith("cuda") else -1

        # Preload multiple detectors with different det_size
        self.detectors = {
            "small": FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            "medium": FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            "large": FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
        }

        self.detectors["small"].prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.detectors["medium"].prepare(ctx_id=ctx_id, det_size=(960, 960))
        self.detectors["large"].prepare(ctx_id=ctx_id, det_size=(1280, 1280))

        self.embeddings_by_label = {}
        self.to_pil = ToPILImage()

    def load_image(self, img_path):
        """Loads an image from a path using OpenCV."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Unable to read image: {img_path}")
        return img

    def select_detector(self, img):
        """Chooses the right detector based on image size."""
        h, w = img.shape[:2]
        if max(h, w) <= 720:
            return self.detectors["small"]
        elif max(h, w) <= 1280:
            return self.detectors["medium"]
        else:
            return self.detectors["large"]

    def detect_faces(self, img):
        """Detects faces in an image using adaptive det_size."""
        detector = self.select_detector(img)
        faces = detector.get(img)
        return faces

    def draw_faces(self, img, faces):
        """Draws bounding boxes and landmarks on faces."""
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
        """Extracts embeddings from a list of detected faces."""
        embeddings = []
        for face in faces:
            emb = torch.tensor(face.embedding)
            embeddings.append(emb)
        return embeddings

    def process_image(self, img, label, display=True):
        """
        Processes a single image (either from a path or as a tensor).
         - Loads image or tensor
         - Detects faces and selects the largest face
         - Gets embedding for the selected face
         - Optionally displays the image with drawn box
        """
        # If input is a file path, load the image
        if isinstance(img, (str, Path)):
            try:
                img_orig = self.load_image(img)
            except ValueError as e:
                print(e)
                return
        elif isinstance(img, torch.Tensor):
            # Denormalize tensor, convert to NumPy array
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img * std + mean
            img_tensor = (img_tensor * 255).byte().permute(1, 2, 0)
            img_orig = img_tensor.cpu().numpy()
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(
                "Input 'img' must be a file path or a torch.Tensor."
            )

        faces = self.detect_faces(img_orig)

        if not faces:
            print(f"No face detected for label: {label}")
            if display:
                img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 6))
                plt.imshow(img_rgb)
                plt.axis("off")
                plt.title(f"No Face Detected: {label}")
                plt.show()
            return

        # Select the largest face
        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1]),
        )
        drawn_img = self.draw_faces(img_orig, [largest_face])

        if display:
            img_rgb = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(f"Detected Largest Face: {label}")
            plt.show()

        embedding = torch.tensor(largest_face.embedding)
        self.embeddings_by_label.setdefault(label, []).append(embedding)

    def process_directory(self, directory, display=True):
        directory = Path(directory)
        image_files = get_image_files(directory)
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                label = img_path.stem.replace("_", " ")
                self.process_image(img_path, label, display=display)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    def process_tensors(self, tensor_dict, display=False):
        print("Processing tensors from MovieDataProcessor...")
        for actor_id, tensor in tqdm(
            tensor_dict.items(), desc="Processing tensors"
        ):
            try:
                self.process_image(tensor, str(actor_id), display=display)
            except Exception as e:
                print(f"Error processing tensor for actor ID {actor_id}: {e}")

    def average_embeddings(self):
        return {
            label: torch.stack(embs).mean(0)
            for label, embs in self.embeddings_by_label.items()
            if embs
        }

    def get_embeddings(self):
        return self.average_embeddings()

    def save_embeddings(self, filename="actor_embeddings.pth"):
        averaged = self.average_embeddings()
        torch.save(averaged, filename)
        print(f"Saved embeddings to {filename}")

    def tensor_to_cv2(self, img_tensor: torch.Tensor):
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
        img_tensor = img_tensor.cpu().float() * std + mean
        img_np = (
            img_tensor.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
        )
        img_bgr = np.ascontiguousarray(img_np[:, :, ::-1])
        return img_bgr
