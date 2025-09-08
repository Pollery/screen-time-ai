import requests
import os
import json
import pprint
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import csv
import base64
import pandas as pd
from IPython.display import HTML, display
import torch
from torchvision.transforms import transforms
from PIL import Image


class TMDbClient:
    """
    A client to interact with The Movie Database (TMDb) API.
    """

    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://media.themoviedb.org/t/p/w300_and_h450_bestv2"

    def __init__(self, api_header):
        """
        Initializes the TMDbClient with the API authorization header.
        """
        if not api_header:
            raise ValueError(
                "API header not provided. Please set HEADER environment variable."
            )
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_header}",
        }

    def _make_request(self, endpoint, params=None):
        """
        Helper method to make a GET request to a TMDb endpoint.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request to {url}: {e}")
            return None

    def search_movie(self, query):
        """
        Searches for a movie by its title.
        """
        endpoint = "search/movie"
        params = {
            "query": query,
            "include_adult": False,
            "language": "en-US",
            "page": 1,
        }
        print(f"Searching for movie: {query}...")
        data = self._make_request(endpoint, params)
        if data and data.get("results"):
            print(
                (
                    f"Found movie '{data['results'][0]['title']}' with ID: {data['results'][0]['id']}"
                )
            )
        return data

    def get_movie_credits(self, movie_id):
        """
        Retrieves the cast and crew credits for a specific movie ID.
        """
        endpoint = f"movie/{movie_id}/credits"
        print(f"Retrieving credits for movie ID: {movie_id}...")
        data = self._make_request(endpoint, {"language": "en-US"})
        return data

    def download_actor_images(
        self, actors_list, output_dir
    ):  # Removed default value
        """
        Downloads profile images for a list of actors.
        """
        if not actors_list:
            print("No actors to download images for.")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nStarting image download to '{output_dir}'...")

        # Use tqdm to add a progress bar to the loop
        for actor in tqdm(
            actors_list, desc="Downloading actor images", unit="file"
        ):
            profile_path = actor.get("profile_path")
            if not profile_path:
                print(
                    f"Skipping download for '{actor['name']}' (no profile path found)."
                )
                continue

            file_name = f"{output_dir}/{actor.get('id')}.jpg"

            # Check if the file already exists before downloading
            if os.path.exists(file_name):
                tqdm.write(
                    f"Image for '{actor.get('name')}' already exists. Skipping download."
                )
                continue

            image_url = f"{self.IMAGE_BASE_URL}{profile_path}"
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                with open(file_name, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                tqdm.write(
                    f"Failed to download image for {actor.get('name')}: {e}"
                )

    def save_actors_to_csv(
        self,
        actors_list,
        output_dir,
        file_name,
        images_dir,  # Removed default value
    ):
        """
        Saves actor information to a CSV file including a base64 encoded image.
        """
        if not actors_list:
            print("No actors to save to CSV.")
            return

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)

        print(f"\nSaving actor information to '{file_path}'...")

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["id", "name", "character", "image"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Use tqdm for a progress bar while writing to the CSV
            for actor in tqdm(actors_list, desc="Writing to CSV", unit="row"):
                # Compute the expected image file name based on the actor's name
                image_file = os.path.join(images_dir, f"{actor.get('id')}.jpg")
                encoded_image = ""
                if os.path.exists(image_file):
                    try:
                        with open(image_file, "rb") as img_f:
                            encoded_image = base64.b64encode(
                                img_f.read()
                            ).decode("utf-8")
                    except Exception as e:
                        tqdm.write(
                            f"Error encoding image for {actor.get('name')}: {e}"
                        )
                writer.writerow(
                    {
                        "id": actor.get("id"),
                        "name": actor.get("name"),
                        "character": actor.get("character"),
                        "image": encoded_image,
                    }
                )
        print("CSV file saved successfully!")


class MovieDataProcessor:
    """
    Manages the end-to-end process of fetching movie data,
    downloading actor images, and processing them.
    """

    def __init__(self, client):
        """
        Initializes the processor with a TMDbClient instance.
        """
        self.client = client
        self.images_dir = "images_train"  # This is now just a folder name part
        self.csv_dir = "actors_info"

    @staticmethod
    def filter_actors(credits_data):
        """
        Filters the credits data to extract information about actors,
        skipping those with "(voice)" in their character name.
        """
        actors = []
        if credits_data and credits_data.get("cast"):
            for person in credits_data["cast"]:
                character_name = person.get("character", "")
                if (
                    person.get("known_for_department") == "Acting"
                    and "(voice)" not in character_name
                ):
                    actor_info = {
                        "id": person.get("id"),
                        "name": person.get("name"),
                        "profile_path": person.get("profile_path"),
                        "character": character_name,
                    }
                    actors.append(actor_info)
        return actors

    def process_movie(self, movie_query):
        """
        Orchestrates the entire process for a given movie query.
        """
        movie_data = self.client.search_movie(movie_query)

        if movie_data and movie_data.get("results"):
            first_result_id = movie_data["results"][0]["id"]
            credits_data = self.client.get_movie_credits(first_result_id)
            actors = self.filter_actors(credits_data)

            if not actors:
                print("No actors found for this movie.")
                return None, None

            pprint.pprint(actors)

            # Create a clean directory name from the movie title
            movie_folder_name = (
                movie_query.replace(" ", "_")
                .replace(":", "")
                .replace("/", "")
                .replace("\\", "")
            )
            # Create the full path for the images folder
            images_output_path = os.path.join(
                movie_folder_name, self.images_dir
            )

            # Create a clean filename for the CSV
            movie_filename = movie_folder_name + ".csv"

            # Download images and save CSV
            self.client.download_actor_images(
                actors, images_output_path
            )  # Pass the full path
            self.client.save_actors_to_csv(
                actors,
                self.csv_dir,
                file_name=movie_filename,
                images_dir=images_output_path,
            )

            # Create the DataFrame and tensor dictionary
            actors_info_df = pd.DataFrame(actors)
            actor_tensors = self.get_actor_images_tensors_by_id(
                actors, images_output_path
            )

            # Return both the tensors and the DataFrame
            return actor_tensors, actors_info_df
        else:
            print(f"Could not find a movie for the query: '{movie_query}'")
            return None, None

    def get_actor_images_tensors_by_id(self, actors_list, images_dir):
        print("\nConverting images to PyTorch tensors...")
        tensors_by_id = {}

        # Define the transformations
        to_tensor = transforms.ToTensor()
        center_crop = transforms.CenterCrop(640)

        for actor in tqdm(actors_list, desc="Creating tensors", unit="image"):
            image_file = os.path.join(images_dir, f"{actor.get('id')}.jpg")
            if os.path.exists(image_file):
                try:
                    if os.path.getsize(image_file) > 0:
                        image = Image.open(image_file).convert("RGB")
                        width, height = image.size

                        # Apply crop only if a dimension is larger than 640
                        if width > 640 or height > 640:
                            image = center_crop(image)

                        tensor = to_tensor(image)

                        key = f"{actor.get('id')}"
                        tensors_by_id[key] = tensor
                    else:
                        tqdm.write(
                            f"Skipping empty or corrupted image file: {image_file}"
                        )
                except Exception as e:
                    tqdm.write(
                        f"Error loading image for actor ID {actor.get('id')}: {e}"
                    )

        return tensors_by_id
