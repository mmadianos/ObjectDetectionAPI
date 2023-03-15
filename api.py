from flask import Flask, request, jsonify
import numpy as np
from model import predict_box

# CREATE FLASK APP
app = Flask(__name__)

# CONNECT POST API CALL -> predict_box() function


@app.route('/predict_box', methods=['POST'])
def predict():

    # GET IMAGE REQUEST
    file = request.files['image']

    # SAVE IMAGE AS .JPG
    path = 'images/im-received.jpg'
    file.save(path)

    # PREDICT BOXES
    boxes = predict_box(path)

    return jsonify({'message': 'success', 'box': boxes})


if __name__ == '__main__':
    app.run(debug=True)
