import os
import datetime
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kerastuner as kt


class SkinCancerDetector:
    def __init__(self, train_dir, val_dir, test_dir, log_dir='logs', batch_size=32, model_dir='models',
                 img_size=(180, 180)):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.model_dir = model_dir
        self.model = None
        self.precision = Precision()
        self.recall = Recall()

    def preprocess_data(self):
        train_generator = self.create_data_generator(self.train_dir, augment=False)
        val_generator = self.create_data_generator(self.test_dir)
        test_datagen = self.create_data_generator(self.test_dir)
        return train_generator, val_generator, test_datagen

    def create_data_generator(self, dir=None, augment=False):
        if augment:
            datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2]
            )
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        if dir:
            return datagen.flow_from_directory(
                dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary'
            )
        return datagen

    def quantize_model(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        return tflite_quant_model

    def build_model(self, num_classes=2):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(96, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[
                               'accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')
                           ])

    def train_model(self, train_generator, val_generator, epochs=1000, patience_lr=100, patience_es=100, min_lr=1e-6,
                    min_delta=1e-3, cooldown_lr=20):
        self._check_model()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)
        callbacks = self._create_callbacks(log_dir, current_time, patience_lr, min_lr, min_delta, patience_es,
                                           cooldown_lr)
        history = self.model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)
        return history

    def HParam_tuning(self, train_generator, val_generator, epochs=25):
        def model_builder(hp):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(self.img_size[0], self.img_size[1], 3)))

            # Hyperparameters for the convolutional layers
            for i in range(hp.Int('conv_blocks', 1, 3, default=2)):
                hp_filters = hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32)
                model.add(
                    tf.keras.layers.Conv2D(filters=hp_filters, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
                model.add(tf.keras.layers.Dropout(
                    rate=hp.Float(f'dropout_conv_{i}', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

            model.add(tf.keras.layers.Flatten())

            # Hyperparameters for the dense layers
            for i in range(hp.Int('dense_blocks', 1, 2, default=1)):
                hp_units = hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
                model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
                model.add(tf.keras.layers.Dropout(
                    rate=hp.Float(f'dropout_dense_{i}', min_value=0.0, max_value=0.5, default=0.5, step=0.05)))

            # Output layer
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            # Tuning the learning rate
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            return model

        tuner = kt.Hyperband(model_builder,
                             objective='val_loss',
                             max_epochs=epochs,
                             factor=3,
                             directory='hyperband_logs',
                             seed=42,
                             hyperband_iterations=2,
                             project_name='skin_cancer_detection')

        class ClearTrainingOutput(tf.keras.callbacks.Callback):
            def on_train_end(*args, **kwargs):
                return

        # Adding a callback for TensorBoard
        log_dir = f"logs/hparam_tuning/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        tuner.search(train_generator,
                     epochs=epochs,
                     validation_data=val_generator,
                     callbacks=[ClearTrainingOutput(), tensorboard_callback])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete.
        """)

        # Train the model with the best hyperparameters
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(train_generator,
                       epochs=epochs,
                       validation_data=val_generator,
                       callbacks=[tensorboard_callback])

    def _create_callbacks(self, log_dir, current_time, patience_lr, min_lr, min_delta, patience_es, cooldown_lr):
        tensorboard_callback = TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=0
        )
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=patience_lr, min_lr=min_lr, min_delta=min_delta,
            cooldown=cooldown_lr, verbose=1
        )
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, f"{current_time}_best_model.h5"), save_best_only=True,
            monitor='val_loss', mode='min', verbose=1
        )
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True,
                                                verbose=1)

        return [tensorboard_callback, reduce_lr_callback, model_checkpoint_callback, early_stopping_callback]

    def evaluate_model(self, test_datagen):
        self._check_model()
        test_generator = test_datagen.flow_from_directory(self.test_dir, target_size=self.img_size,
                                                          batch_size=self.batch_size, class_mode='binary')
        test_loss, test_acc, test_precision, test_recall, test_auc = self.model.evaluate(test_generator)
        print(
            f'Test accuracy: {test_acc}, Test precision: {test_precision}, Test recall: {test_recall}, Test AUC: {test_auc}')
        return test_loss, test_acc, test_precision, test_recall, test_auc

    def save_model(self, filename='models/skinvestigator.h5'):
        self._check_model()
        self.model.save(filename)
        tflite_model = self.quantize_model(self.model)
        tflite_model_path = filename.replace('.h5', '-quantized.tflite')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model saved as {filename} and {tflite_model_path}")

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename, custom_objects={"Precision": Precision, "Recall": Recall})
        print(f"Model loaded from {filename}")

    def _check_model(self):
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")
