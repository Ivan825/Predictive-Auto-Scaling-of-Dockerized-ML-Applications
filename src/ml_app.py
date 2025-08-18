from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.json
    # Simulate ML model inference time
    time.sleep(random.uniform(0.05, 0.2))
    
    # Simulate a prediction result
    prediction = {"result": "simulated_defect_detection", "input_data": data}
    
    end_time = time.time()
    response_time = (end_time - start_time) * 1000 # milliseconds
    
    print(f"Request processed in {response_time:.2f} ms")
    return jsonify(prediction)

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
