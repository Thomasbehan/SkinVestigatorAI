import os
import datetime
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skinvestigatorai.core.data_gen import DataGen


class SkinCancerDetector:
    def __init__(self, train_dir, val_dir, test_dir, log_dir='logs'):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.model = None
        self.weight_benign = 1.0
        self.weight_malignant = 1.0
        self.batch_size = 64
        self.augmentations = None
        self.malignant_repeat_count = 1

    def preprocess_data(self):
        self.augmentations = A.Compose([
            A.Rotate(limit=40),
            A.RandomBrightness(),
            A.GaussNoise(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ])

        total_samples = len(os.listdir(self.train_dir + '/benign')) + len(os.listdir(self.train_dir + '/malignant'))

        self.weight_benign = total_samples / (2 * len(os.listdir(self.train_dir + '/benign')))
        self.weight_malignant = total_samples / (2 * len(os.listdir(self.train_dir + '/malignant')))
        print('Benign weight: ', self.weight_benign)
        print('Malignant weight: ', self.weight_malignant)

        train_generator = DataGen(
            self.train_dir,
            batch_size=self.batch_size,
            img_height=150,
            img_width=150,
            augmentations=self.augmentations,
            malignant_repeat_count=self.malignant_repeat_count)

        val_generator = DataGen(
            self.val_dir,
            batch_size=self.batch_size,
            img_height=150,
            img_width=150,
            augmentations=self.augmentations,
            malignant_repeat_count=self.malignant_repeat_count)

        return train_generator, val_generator

    def build_model(self, num_classes):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3),
                                activation='relu',
                                padding='same',
                                input_shape=(150, 150, 3),
                                kernel_initializer='glorot_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3),
                                activation='relu',
                                padding='same',
                                kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(num_classes, activation='softmax', dtype=tf.float32))

        model.compile(optimizer='Adam',
                      loss=BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        self.model = model

    def train_model(self, train_generator, val_generator, epochs=3000):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Create a log directory with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

        # Set up the TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                           update_freq='epoch', profile_batch=0)
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=20,
            min_lr=0.000001,
            cooldown=1,
            min_delta=0.0001,
        )
        # Model checkpoint name unique to time and size of the model
        model_name = 'model-{epoch:03d}-{val_accuracy:.4f}-{val_loss:.4f}.h5'

        model_checkpoint_callback = ModelCheckpoint(filepath="models/v2/" + model_name,
                                                    save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight={0: self.weight_benign, 1: self.weight_malignant},
            callbacks=[tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback])
        return history

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        test_generator = DataGen(
            self.test_dir,
            batch_size=self.batch_size,
            img_height=150,
            img_width=150,
            augmentations=self.augmentations,
            malignant_repeat_count=self.malignant_repeat_count)

        test_loss, test_acc = self.model.evaluate(test_generator)
        print('Test accuracy:', test_acc)
        return test_loss, test_acc

    def save_model(self, filename='models/skinvestigator_acc.h5'):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        self.model.save(filename)
        self.model.summary()
