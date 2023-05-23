import numpy as np
from PIL import Image
from pyramid.view import view_config
from pyramid.httpexceptions import HTTPBadRequest
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model

model_to_run = {'nano': 'models/skinvestigator_nano_40MB_91_38_acc.h5',
                 'basic': 'models/skin_cancer_detection_model.h5',
                 'best': 'models/v2/best_model.h5',
                 'gpu': 'skin_cancer_detection_model_all_GPU.h5',}
model = load_model(model_to_run['best'])

# Define the class labels
class_labels = ['benign', 'malignant', 'unknown']


@view_config(route_name='predict', request_method='POST', renderer='json')
def predict_view(request):
    # Read the image file from the request
    image_file = request.POST['image'].file

    try:
        # Open and preprocess the image
        image = Image.open(image_file).convert('RGB')
        image = image.resize((150, 150))
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions)]

        # Return the prediction result
        return {
            'prediction': predicted_class,
            'confidence': float(predictions[0][np.argmax(predictions)]) * 100
        }
    except Exception as e:
        return HTTPBadRequest(reason=str(e))


@view_config(route_name='dashboard', renderer='skinvestigatorai:templates/dashboard.jinja2')
def dashboard_view(request):
    if 'prediction' in request.session:
        prediction = request.session['prediction']
        confidence = request.session['confidence']
        del request.session['prediction']
        del request.session['confidence']
    else:
        prediction = None
        confidence = None

    return {
        'prediction': prediction,
        'confidence': confidence
    }
