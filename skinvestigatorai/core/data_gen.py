import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, img_height, img_width, augmentations, is_minority_class=False, malignant_repeat_count=1):
        self.directory = directory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.malignant_repeat_count = malignant_repeat_count
        self.augment = augmentations
        self.is_minority_class = is_minority_class
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.generator = self.datagen.flow_from_directory(directory,
                                                          target_size=(self.img_height, self.img_width),
                                                          batch_size=self.batch_size,
                                                          class_mode='categorical')

    @property
    def class_indices(self):
        return self.generator.class_indices

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        x, y = self.generator[index]
        # Apply more augmentations if this is the minority class
        if self.is_minority_class:
            print(self.malignant_repeat_count)
            return np.stack([self.augment(image=i)["image"] for i in x] * self.malignant_repeat_count, axis=0), np.repeat(y, self.malignant_repeat_count, axis=0)
        else:
            return np.stack([self.augment(image=i)["image"] for i in x], axis=0), y
