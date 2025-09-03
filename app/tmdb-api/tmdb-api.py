import requests
import os
import json
import pprint
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import csv


class TMDbClient:
    """
    A client to interact with The Movie Database (TMDb) API.

    This class encapsulates API calls for searching movies, retrieving
    credits, and downloading actor profile images.
    """

    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://media.themoviedb.org/t/p/w300_and_h450_bestv2"

    def __init__(self, api_header):
        """
        Initializes the TMDbClient with the API authorization header.

        Args:
            api_header (str): The Authorization header token from TMDb.
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

        Args:
            endpoint (str): The API endpoint path.
            params (dict, optional): URL parameters for the request. Defaults to None.

        Returns:
            dict: The JSON response as a dictionary, or None on failure.
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

        Args:
            query (str): The movie title to search for.

        Returns:
            dict or None: A dictionary containing the movie data, or None on failure.
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

        Args:
            movie_id (int): The ID of the movie.

        Returns:
            dict or None: A dictionary containing the credits data, or None on failure.
        """
        endpoint = f"movie/{movie_id}/credits"
        print(f"Retrieving credits for movie ID: {movie_id}...")
        data = self._make_request(endpoint, {"language": "en-US"})
        return data

    @staticmethod
    def get_actors_from_credits(credits_data):
        """
        Filters the credits data to extract information about actors.

        Args:
            credits_data (dict): The credits data from the TMDb API.

        Returns:
            list: A list of dictionaries, where each dictionary contains actor info.
        """
        actors = []
        if credits_data and credits_data.get("cast"):
            for person in credits_data["cast"]:
                if person.get("known_for_department") == "Acting":
                    actor_info = {
                        "id": person.get("id"),
                        "name": person.get("name"),
                        "profile_path": person.get("profile_path"),
                        "character": person.get("character"),
                    }
                    actors.append(actor_info)
        return actors

    def download_actor_images(self, actors_list, output_dir="images_train"):
        """
        Downloads profile images for a list of actors.

        Args:
            actors_list (list): A list of dictionaries with actor information.
            output_dir (str, optional): The directory to save the images. Defaults to "images_train".
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

            file_name = (
                f"{output_dir}/{actor.get('name').replace(' ', '_')}.jpg"
            )

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
        self, actors_list, output_dir="actors_info", file_name="actors.csv"
    ):
        """
        Saves actor information to a CSV file.

        Args:
            actors_list (list): A list of dictionaries with actor information.
            output_dir (str, optional): The directory to save the CSV file. Defaults to "actors_info".
            file_name (str, optional): The name of the CSV file. Defaults to "actors.csv".
        """
        if not actors_list:
            print("No actors to save to CSV.")
            return

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)

        print(f"\nSaving actor information to '{file_path}'...")

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["id", "name", "character"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Use tqdm for a progress bar while writing to the CSV
            for actor in tqdm(actors_list, desc="Writing to CSV", unit="row"):
                writer.writerow(
                    {
                        "id": actor.get("id"),
                        "name": actor.get("name"),
                        "character": actor.get("character"),
                    }
                )
        print("CSV file saved successfully!")


if __name__ == "__main__":
    # Load environment variables
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
    HEADER = os.getenv("HEADER")

    if not HEADER:
        print("Error: HEADER environment variable is not set.")
        exit()

    # Create a client instance
    client = TMDbClient(api_header=HEADER)

    # Main script logic
    movie_query = "harry potter and the deathly hallows part 1"

    movie_data = client.search_movie(movie_query)

    if movie_data and movie_data.get("results"):
        first_result_id = movie_data["results"][0]["id"]

        credits_data = client.get_movie_credits(first_result_id)

        actors = client.get_actors_from_credits(credits_data)

        pprint.pprint(actors)

        client.download_actor_images(actors)

        client.save_actors_to_csv(actors)
    else:
        print(f"Could not find a movie for the query: '{movie_query}'")
