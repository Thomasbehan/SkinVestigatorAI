import os
import shutil
from unittest.mock import patch
from skinvestigatorai.core.data_scraper import DataScraper


def test_create_output_folders():
    data_dir = "test_data/"
    test_dir = "test_data/test"
    train_dir = "test_data/train"
    temp_dir = "test_data/temp"
    benign_dir = "test_data/temp/benign"
    malignant_dir = "test_data/temp/malignant"

    data_scraper = DataScraper(data_dir, 1)

    # Call the internal function
    data_scraper._create_output_folders()

    # Check if the directories were created
    assert os.path.exists(test_dir)
    assert os.path.exists(train_dir)
    assert os.path.exists(temp_dir)
    assert os.path.exists(benign_dir)
    assert os.path.exists(malignant_dir)

    # Cleanup
    shutil.rmtree("test_data")


@patch("skinvestigatorai.core.data_scraper.DataScraper.download_and_split_images")
def test_download_images(mock_download_images):
    data_scraper = DataScraper()
    data_scraper.download_and_split_images()

    # Test if the download_images() method is called once
    mock_download_images.assert_called_once()

