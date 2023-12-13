from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# 载入模型
model = load_model('mnist_convnet.h5')

@app.route('/predict', methods=['POST'])
def predict():
    print(f"Received connection from: {request.remote_addr}")
    data = request.get_json(force=True)
    prediction = model.predict(np.array([data['image']]))
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000)
