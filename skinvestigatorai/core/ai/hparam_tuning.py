import os
from skinvestigatorai.core.ai.detector import SkinCancerDetector
from skinvestigatorai.core.ai.config import train_dir, val_dir, test_dir
from skinvestigatorai.core.data_scraper import DataScraper


def main():
    # check if data is downloaded and if not download it
    if not os.path.exists(train_dir):
        print('Downloading data...')
        downloader = DataScraper()
        downloader.download_and_split_images()
        print('Done downloading data')

    # Print count of files in each directory
    print('Train:', len(os.listdir(train_dir + '/benign')), 'benign,', len(os.listdir(train_dir + '/malignant')),
          'malignant')
    # Print count of files in each directory
    print('Test:', len(os.listdir(test_dir + '/benign')), 'benign,', len(os.listdir(test_dir + '/malignant')),
          'malignant')

    detector = SkinCancerDetector(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_datagen = detector.preprocess_data()
    detector.build_model(num_classes=len(train_generator.class_indices))
    detector.HParam_tuning(train_generator, val_generator)


if __name__ == '__main__':

    if not os.path.exists('data/train/benign'):
        downloader = DataScraper()
        print('Done training models')
        print('Training model with all data')
        downloader.download_and_split_images()
    main()
