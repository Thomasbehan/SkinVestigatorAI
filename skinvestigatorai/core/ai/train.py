import os
from skinvestigatorai.core.ai.detector import SkinCancerDetector
from skinvestigatorai.core.ai.config import train_dir, val_dir, test_dir
from skinvestigatorai.core.data_scraper import DataScraper


def main(filename='models/v2/skinvestigator.h5'):
    # check if data is downloaded and if not download it
    if not os.path.exists(train_dir):
        print('Downloading data...')
        downloader = DataScraper()
        downloader.download_images(-1)
        print('Done downloading data')

    # Print count of files in each directory
    print('Train:', len(os.listdir(train_dir + '/benign')), 'benign,', len(os.listdir(train_dir + '/malignant')),
          'malignant')

    detector = SkinCancerDetector(train_dir, val_dir, test_dir)
    train_generator, val_generator = detector.preprocess_data()
    detector.build_model(num_classes=len(train_generator.class_indices))
    detector.train_model(train_generator, val_generator)
    detector.evaluate_model()
    detector.save_model(filename)


if __name__ == '__main__':

    if not os.path.exists('data/train/benign'):
        downloader = DataScraper()
        print('Done training models')
        print('Training model with all data')
        downloader.download_images(-1)
    main('skin_cancer_detection_model_all_GPU.h5')
