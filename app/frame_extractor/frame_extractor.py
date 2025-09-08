import cv2
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image


class VideoProcessor:
    """
    Manages the video conversion, analysis, and frame extraction process.
    """

    def __init__(self, input_video_path, output_directory):
        """
        Initializes the VideoProcessor.

        Args:
            input_video_path (str): The path to the input video file.
            output_directory (str): The directory for saving output files.
        """
        self.input_video_path = Path(input_video_path)
        self.output_directory = Path(output_directory)
        self.converted_video_path = (
            self.output_directory / "converted_video.mp4"
        )
        self.frame_extractor = FrameExtractor(
            str(self.converted_video_path), str(self.output_directory)
        )

    def convert_video(self):
        """
        Converts the input video to H.264 format using GPU acceleration.
        """
        self.output_directory.mkdir(parents=True, exist_ok=True)
        conversion_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(self.input_video_path),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "fast",
            "-crf",
            "23",
            str(self.converted_video_path),
        ]

        print("Starting video conversion...")
        conv_start = time.perf_counter()
        try:
            subprocess.run(conversion_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during video conversion: {e}")
            sys.exit(1)
        conv_end = time.perf_counter()
        print(f"Conversion complete: {self.converted_video_path}")
        print(f"Video conversion took {conv_end - conv_start:.2f} seconds.\n")

    def analyze_video(self):
        """
        Analyzes the converted video and prints its properties.
        """
        cap = cv2.VideoCapture(str(self.converted_video_path))
        if not cap.isOpened():
            print("Error: Could not open converted video file for analysis.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_seconds = total_frames / fps if fps != 0 else 0
        expected_images = int(total_seconds)
        approx_image_size = 200 * 1024  # 200 KB
        total_expected_space_MB = (expected_images * approx_image_size) / (
            1024 * 1024
        )

        print(f"Video duration: {total_seconds:.2f} seconds")
        print(f"Expected number of images to be extracted: {expected_images}")
        print(
            f"Estimated total space for images: {total_expected_space_MB:.2f} MB"
        )
        cap.release()

    def process(self, extract_frames=True):
        """
        Runs the full video processing workflow.

        Args:
            extract_frames (bool): If True, proceeds with frame extraction.
        """
        self.convert_video()
        self.analyze_video()

        # if extract_frames:
        #     choice = (
        #         input("Do you want to continue with frame extraction? (y/n): ")
        #         .strip()
        #         .lower()
        #     )
        #     if choice == "y":
        return self.frame_extractor.extract_frames_per_second()
        #     else:
        #         print("Frame extraction aborted.")
        #         return None
        # return None


class FrameExtractor:
    """
    A class to extract frames from a video file and convert them to a PyTorch tensor.
    """

    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder
        self.cap = None

    def _create_output_folder(self):
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder '{self.output_folder}' ensured.")

    def _open_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file at {self.video_path}")
            return False
        return True

    def extract_frames_per_second(self):
        """
        Extracts one frame per second and returns a PyTorch tensor dictionary.
        """
        if not self._open_video():
            return None

        self._create_output_folder()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(round(fps))
        frame_count = 0
        saved_frame_count = 0

        # New: Use a dictionary to store frames by number
        extracted_frames_dict = {}

        print(f"Video FPS: {fps:.2f}")
        print(f"Extracting a frame every {frames_to_skip} frames...")

        start_time = time.perf_counter()

        # Define a transformation to convert the PIL Image to a PyTorch tensor
        # This will be used for every frame
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (640, 640)
                ),  # Ensure frames match the model's expected input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % frames_to_skip == 0:
                output_filename = os.path.join(
                    self.output_folder, f"frame_{saved_frame_count:03d}.jpg"
                )
                cv2.imwrite(output_filename, frame)
                # print(f"Saved {output_filename}") # Removed to avoid cluttering output

                # Convert the BGR frame to a PyTorch tensor
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Apply the transformation to the PIL Image
                tensor_frame = transform(pil_image)

                # Store the tensor in the dictionary with the frame count as key
                extracted_frames_dict[saved_frame_count] = tensor_frame

                saved_frame_count += 1
            frame_count += 1

        self.cap.release()
        end_time = time.perf_counter()
        extraction_time = end_time - start_time

        print("\nFrame extraction complete!")
        print(f"Total frames extracted: {saved_frame_count}")
        print(f"Extraction took {extraction_time:.2f} seconds.")

        # Return the dictionary of tensors
        if extracted_frames_dict:
            print(
                f"Returned a dictionary with {len(extracted_frames_dict)} frame tensors."
            )
            return extracted_frames_dict
        else:
            print("No frames were extracted. Returning an empty dictionary.")
            return {}


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    video_file = (
        current_dir.parent.parent
        / "data"
        / "Harry Potter and the Deathly Hallows - Main Trailer [Su1LOpjvdZ4].webm"
    )
    output_directory = video_file.stem

    # Initialize and run the video processor
    processor = VideoProcessor(video_file, output_directory)
    video_tensors_dict = processor.process()

    # You can now use the video_tensors_dict with FaceEmbeddingGenerator.
    if video_tensors_dict:
        # Example of how to access a tensor
        first_frame_id = list(video_tensors_dict.keys())[0]
        print(
            f"Tensor for frame {first_frame_id} has shape: {video_tensors_dict[first_frame_id].shape}"
        )
    else:
        print("No video tensors were generated.")
