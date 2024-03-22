<img src="/skinvestigatorai/static/logo.png" align="right" width="100" height="100" />

# SkinVestigatorAI  ![View SkinVestigatorAI on GitHub](https://img.shields.io/github/stars/Thomasbehan/SkinVestigatorAI?color=232323&label=SkinVestigatorAI&logo=github&labelColor=232323)
![Sensitivity Score](https://img.shields.io/badge/Sensitivity-0.84035-blue)
![Specificity Score](https://img.shields.io/badge/Specificity-0.84019-blue)
![Precision Score](https://img.shields.io/badge/Precision-0.84035-blue)
![F1 Score](https://img.shields.io/badge/F1-0.84467-blue)
![Accuracy Score](https://img.shields.io/badge/Accuracy-0.84035-blue)
![Loss Score](https://img.shields.io/badge/Loss-0.23201-blue)
![AUC Score](https://img.shields.io/badge/AUC-0.91692-blue)
![GitHub license](https://img.shields.io/github/license/Thomasbehan/SkinVestigatorAI) [![Actions Status](https://github.com/Thomasbehan/SkinVestigatorAI/workflows/Automated%20Testing/badge.svg)](https://github.com/Thomasbehan/SkinVestigatorAI/actions)
[![Actions Status](https://github.com/Thomasbehan/SkinVestigatorAI/workflows/CodeQL/badge.svg)](https://github.com/Thomasbehan/SkinVestigatorAI/actions)

> SkinVestigatorAI is an open-source project for deep learning-based skin cancer detection. It aims to create a reliable tool and foster community involvement in critical AI problems. The repository includes code for data preprocessing, model building, and performance evaluation. Contribute and shape the future of skin cancer detection.

[![Demo](https://img.shields.io/badge/-Live_Demo-black?style=for-the-badge&logo=render)](https://skinvestigator.onrender.com/) 

<sub><i>Please note that the application enters a dormant state when not in use to conserve resources. This means it might take a moment to warm up when you first access the site. Any initial slow down will ease after a moment. Thank you for your patience.
</i></sub>

## Table of Contents
- [Getting Started](#getting-started)
- [Data](#data)
- [Model](#model)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Citation](#citation)
- [Disclaimer](#disclaimer)

# Getting Started
<img src="screenshot.png" align="center" />

These instructions will help you set up the project on your local machine for development and testing purposes. See [deployment](#deployment) for notes on deploying the project on a live system.

## Quick Setup for Development

To quickly set up SkinVestigatorAI for development, follow these steps
(Requires Python >=3.11):

1. **Upgrade Your Packaging Tools:**
   Ensure your `pip` and `setuptools` are up-to-date by running:
   ```bash
   python -m pip install --upgrade pip setuptools
   ```

2. **Install SkinVestigatorAI:**
   In the project directory, install the project in editable mode with:
   ```bash
   python -m pip install -e .
   ```

3. **Run the Application:**
   Start the application with auto-reloading using:
   ```bash
   pserve development.ini --reload
   ```

## Running the Tests and Linting
[![Actions Status](https://github.com/Thomasbehan/SkinVestigatorAI/workflows/Automated%20Testing/badge.svg)](https://github.com/Thomasbehan/SkinVestigatorAI/actions)
[![Actions Status](https://github.com/Thomasbehan/SkinVestigatorAI/workflows/CodeQL/badge.svg)](https://github.com/Thomasbehan/SkinVestigatorAI/actions)

### Running the Tests
To run the tests, run the following command:
```bash
python -m pytest
```

### Running the Linter
To run the linter, run the following command:
```bash
python -m ruff check
```

## Data
The DataScraper tool within this application is designed to download and preprocess skin lesion images from the ISIC Archive for use in machine learning projects. The images
are stored in three separate directories for training, validation, and testing, featuring a total of 40,194 images. This substantial dataset aims to provide a comprehensive basis for accurate skin lesion analysis and classification.

The data is organised as follows:
- Train: 32,155 images
- Test: 8,039 images

### Data Source
The dataset used for training the model is sourced from the International Skin Imaging Collaboration (ISIC) Archive. The ISIC Archive is a large-scale resource for skin image analysis, providing open access to a wide variety of images for the development and evaluation of automated diagnostic systems.

For more information about the ISIC Archive and to access the data, visit [ISIC Archive](https://www.isic-archive.com).

### Data Organization
The images are organized into three folders:

1. `data/train`: Contains 80% of the total images, which are used for training the model.
2. `data/test`: Contains 20% of the total images, used for testing the model's performance during and after training.

## Model
The `SkinCancerDetector` model employs a sophisticated deep learning architecture tailored for the accurate classification of skin lesions as benign or malignant. Built on TensorFlow, the model features a sequential arrangement of layers, utilising convolutional neural networks (CNNs) for their powerful image processing capabilities.

### Architecture Overview
The architecture is meticulously designed to capture the intricate patterns and features of skin lesions through multiple stages of convolutional layers, each followed by max pooling to reduce spatial dimensions and dropout layers to prevent overfitting. The model's core is structured as follows:

- **Convolutional Layers:** Multiple layers with ReLU activation to extract features from images.
- **Max Pooling Layers:** Applied after convolutional layers to reduce the size of the feature maps, thereby reducing the number of parameters and computation in the network.
- **Dropout Layers:** Used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
- **Dense Layers:** Fully connected layers that learn non-linear combinations of the high-level features extracted by the convolutional layers.
- **Output Layer:** A dense layer with a sigmoid activation function to classify the input image as benign or malignant.


```bash
   Model: "sequential"
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #
   =================================================================
   conv2d (Conv2D)              (None, 180, 180, 128)     1280
   _________________________________________________________________
   max_pooling2d (MaxPooling2D) (None, 90, 90, 128)       0
   _________________________________________________________________
   dropout (Dropout)            (None, 90, 90, 128)       0
   _________________________________________________________________
   conv2d_1 (Conv2D)            (None, 90, 90, 256)       295168
   _________________________________________________________________
   max_pooling2d_1 (MaxPooling2 (None, 45, 45, 256)       0
   _________________________________________________________________
   dropout_1 (Dropout)          (None, 45, 45, 256)       0
   _________________________________________________________________
   conv2d_2 (Conv2D)            (None, 45, 45, 192)       442560
   _________________________________________________________________
   max_pooling2d_2 (MaxPooling2 (None, 22, 22, 192)       0
   _________________________________________________________________
   dropout_2 (Dropout)          (None, 22, 22, 192)       0
   _________________________________________________________________
   flatten (Flatten)            (None, 92416)             0
   _________________________________________________________________
   dense (Dense)                (None, 64)                5914688
   _________________________________________________________________
   dropout_3 (Dropout)          (None, 64)                0
   _________________________________________________________________
   dense_1 (Dense)              (None, 96)                6240
   _________________________________________________________________
   dropout_4 (Dropout)          (None, 96)                0
   _________________________________________________________________
   dense_2 (Dense)              (None, 1)                 97
   =================================================================
   Total params: 6,660,033
   Trainable params: 6,660,033
   Non-trainable params: 0
   _________________________________________________________________
```

### Training and Optimization
The model is compiled with the Adam optimizer and binary cross-entropy loss function, which are well-suited for binary classification tasks. It leverages metrics such as accuracy, precision, recall, and AUC to evaluate performance throughout the training process.

Training involves the use of a data generator for efficient handling of large image datasets, augmenting the training data to improve generalization. The model also incorporates callbacks for early stopping, learning rate reduction on plateau, and model checkpointing to save the best-performing model.

This advanced architecture and training regimen enable the `SkinCancerDetector` to achieve high accuracy in distinguishing between benign and malignant skin lesions, making it a valuable tool for aiding in the early detection of skin cancer.


## Performance
The updated model demonstrates significant improvements in its ability to classify skin lesions accurately, achieving an accuracy of 84% and a loss of 0.23 on the testing dataset. The model's sensitivity, specificity, precision, and F1 score have also seen considerable enhancements, with the following scores reported on the testing dataset:

- Sensitivity: 84.035%
- Specificity: 84.019%
- Precision: 84.035%
- F1 Score: 84.467%
- Accuracy: 84.035%
- Loss: 0.23201
- AUC: 91.692%

## Contributing
We encourage contributions to SkinVestigatorAI! For guidelines on contributing, please read [CONTRIBUTING.md](CONTRIBUTING.md). By participating in this project, you agree to abide by its terms.

## License
SkinVestigatorAI is released under the GNU General Public License v3.0. For more details, see the [LICENSE.md](LICENSE.md) file.

## Acknowledgments
We extend our gratitude to the International Skin Imaging Collaboration (ISIC) for providing access to their extensive archive of skin lesion images, which has been instrumental in the development and refinement of our model.

## References
- International Skin Imaging Collaboration (ISIC). The ISIC Archive. https://www.isic-archive.com

## Citation
For academic and research use, please cite our work as follows:

"SkinVestigator: A Deep Learning-Based Skin Cancer Detection Tool, available at: https://github.com/Thomasbehan/SkinVestigatorAI", 2024.


## Disclaimer
SkinVestigatorAI is not intended for clinical diagnosis or medical use. It is a research tool aimed at fostering developments in the field of automated skin lesion analysis. Always consult a qualified healthcare provider for medical advice and diagnosis.

