import os
import argparse
import requests
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class DataScraper:
    def __init__(self, output_dir="data", max_pages=-1):
        self.output_dir = output_dir
        self.api_url = "https://api.isic-archive.com/api/v2/images"
        self.failed_downloads_path = os.path.join(self.output_dir, "failed_downloads.txt")
        self.max_pages = max_pages
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _create_output_folders(self):
        for category in ["train", "test", "temp"]:
            for label in ["benign", "malignant"]:
                os.makedirs(os.path.join(self.output_dir, category, label), exist_ok=True)
        os.makedirs(os.path.dirname(self.failed_downloads_path), exist_ok=True)

    def _download_image(self, image_data):
        image_url, file_path = image_data
        if os.path.exists(file_path):
            return file_path, True  # Indicates success to avoid marking as failed
        try:
            response = self.session.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path, True
            else:
                return image_url, False
        except requests.RequestException:
            return image_url, False

    def _split_data(self, images, test_size=0.2):
        train, test = train_test_split(images, test_size=test_size, random_state=42)
        return train, test

    def _move_images(self, images, source_folder, dest_folder):
        for image in images:
            shutil.move(os.path.join(source_folder, image), os.path.join(dest_folder, image))

    def download_and_split_images(self):
        self._create_output_folders()
        temp_folder = os.path.join(self.output_dir, "temp")
        failed_downloads = set()
        if os.path.exists(self.failed_downloads_path):
            with open(self.failed_downloads_path) as f:
                failed_downloads = {line.strip() for line in f}

        next_url = self.api_url
        params = {'limit': 100, 'offset': 0}
        total_images_downloaded = 0
        page_count = 0

        while next_url and (self.max_pages == -1 or page_count < self.max_pages):
            response = self.session.get(next_url, params=params)
            if response.status_code != 200:
                break

            data = response.json()
            next_url = data.get("next", None)
            page_count += 1

            download_tasks = []
            for image in data['results']:
                isic_id = image['isic_id']
                image_url = image['files']['full']['url']
                benign_malignant = image['metadata']['clinical'].get('benign_malignant', 'unknown')
                if benign_malignant in ['benign', 'malignant'] and image_url not in failed_downloads:
                    file_path = os.path.join(temp_folder, benign_malignant, f"{isic_id}.jpg")
                    download_tasks.append((image_url, file_path))

            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_url = {executor.submit(self._download_image, task): task for task in download_tasks}
                for future in as_completed(future_to_url):
                    url, success = future.result()
                    if success:
                        total_images_downloaded += 1
                    else:
                        with open(self.failed_downloads_path, 'a') as f:
                            f.write(f"{url}\n")

            print(f"Total images downloaded: {total_images_downloaded}")

        for label in ['benign', 'malignant']:
            images = [img for img in os.listdir(os.path.join(temp_folder, label)) if img.endswith(".jpg")]
            train_images, test_images = self._split_data(images)
            self._move_images(train_images, os.path.join(temp_folder, label),
                              os.path.join(self.output_dir, "train", label))
            self._move_images(test_images, os.path.join(temp_folder, label),
                              os.path.join(self.output_dir, "test", label))

        shutil.rmtree(temp_folder)
        print("Images split into train and test sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images from ISIC Archive and split into training and testing sets.")
    parser.add_argument("-p", "--pages", type=int, default=-1,
                        help="Number of pages to download. Default is -1, which downloads all pages.")
    args = parser.parse_args()

    scraper = DataScraper(max_pages=args.pages)
    scraper.download_and_split_images()
