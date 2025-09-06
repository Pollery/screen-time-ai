import cv2
import os
import subprocess
import sys
import time
from pathlib import Path


class FrameExtractor:
    """
    A class to extract frames from a video file based on a specified interval.
    """

    def __init__(self, video_path, output_folder):
        """
        Initializes the FrameExtractor with the video path and output folder.

        Args:
            video_path (str): The path to the input video file.
            output_folder (str): The directory to save the extracted frames.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.cap = None

    def _create_output_folder(self):
        """
        Creates the output folder if it doesn't already exist.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder '{self.output_folder}' ensured.")

    def _open_video(self):
        """
        Opens the video file and returns True if successful, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file at {self.video_path}")
            return False
        return True

    def extract_frames_per_second(self):
        """
        Extracts one frame per second from the video.
        """
        if not self._open_video():
            return

        self._create_output_folder()

        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        # The number of frames to skip to get one frame per second
        frames_to_skip = int(round(fps))

        frame_count = 0
        saved_frame_count = 0

        print(f"Video FPS: {fps:.2f}")
        print(f"Extracting a frame every {frames_to_skip} frames...")

        start_time = time.perf_counter()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % frames_to_skip == 0:
                output_filename = os.path.join(
                    self.output_folder, f"frame_{saved_frame_count:03d}.jpg"
                )
                cv2.imwrite(output_filename, frame)
                print(f"Saved {output_filename}")
                saved_frame_count += 1

            frame_count += 1

        self.cap.release()
        end_time = time.perf_counter()
        extraction_time = end_time - start_time

        print("\nFrame extraction complete!")
        print(f"Total frames extracted: {saved_frame_count}")
        print(f"Extraction took {extraction_time:.2f} seconds.")


if __name__ == "__main__":

    # Determine the current directory and adjust path accordingly
    current_dir = Path(__file__).resolve().parent
    video_file = (
        current_dir.parent.parent
        / "data"
        / "Harry Potter and the Deathly Hallows - Main Trailer [Su1LOpjvdZ4].webm"
    )
    output_directory = (
        video_file.stem
    )  # output directory named after the video filename

    # Define a converted video file path
    converted_video_file = (
        current_dir.parent.parent / "data" / "converted_video.mp4"
    )

    # Convert the original video to H.264 (using GPU acceleration)
    conversion_command = [
        "ffmpeg",
        "-y",  # overwrite output file if it exists
        "-i",
        str(video_file),
        "-c:v",
        "h264_nvenc",  # use NVIDIA NVENC for GPU accelerated encoding
        "-preset",
        "fast",
        "-crf",
        "23",
        str(converted_video_file),
    ]

    print("Starting video conversion...")
    conv_start = time.perf_counter()
    try:
        subprocess.run(conversion_command, check=True)
    except subprocess.CalledProcessError:
        print("Error during video conversion.")
        sys.exit(1)
    conv_end = time.perf_counter()
    conv_time = conv_end - conv_start
    print(f"Conversion complete: {converted_video_file}")
    print(f"Video conversion took {conv_time:.2f} seconds.\n")

    # Open the converted video to calculate expected extraction details
    cap = cv2.VideoCapture(str(converted_video_file))
    if cap.isOpened():
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_seconds = total_frames / fps if fps != 0 else 0
        expected_images = int(total_seconds)
        # Assuming an approximate size of 200 KB per JPEG image
        approx_image_size = 200 * 1024
        total_expected_space_MB = (expected_images * approx_image_size) / (
            1024 * 1024
        )
        print(f"Video duration: {total_seconds:.2f} seconds")
        print(f"Expected number of images to be extracted: {expected_images}")
        print(
            f"Estimated total space for images: {total_expected_space_MB:.2f} MB"
        )
    else:
        print("Error: Could not open converted video file for analysis.")
    cap.release()

    # Ask if the user wants to continue with extraction
    choice = (
        input("Do you want to continue with frame extraction? (y/n): ")
        .strip()
        .lower()
    )
    if choice == "y":
        extractor = FrameExtractor(str(converted_video_file), output_directory)
        extractor.extract_frames_per_second()
    else:
        print("Frame extraction aborted.")
