from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skinvestigatorai.core.custom_image_data_generator import CustomImageDataGenerator


def test_custom_image_data_generator():
    # Create a custom image data generator
    custom_image_data_gen = CustomImageDataGenerator()

    # Test that the generator is an instance of ImageDataGenerator
    assert isinstance(custom_image_data_gen, ImageDataGenerator)

    # Test that the overridden method is present
    assert hasattr(custom_image_data_gen, '_get_batches_of_transformed_samples')


