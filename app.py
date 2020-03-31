# Flask
from flask import Flask , request, render_template,  jsonify
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5003/')

#add your model_path
model_path = '/home/chirag/ML_Model/mobile.h5'

# Load your own trained model
model = load_model(model_path)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))
        pred_class = decode_predictions(preds, top=1)

        result = str(pred_class[0][0][1])
        # result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    app.run(port=5003, threaded=False)
