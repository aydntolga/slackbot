import logging
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
import requests
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
APP_TOKEN = os.getenv("APP_TOKEN")
ML_MODEL_PATH = "maps_updated.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


try:
    pipelineMaps = joblib.load(ML_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    logging.info("Model and Label Encoder successfully loaded.")
except Exception as e:
    logging.error(f"Error loading model or label encoder: {e}")

bolt_app = App(token=SLACK_BOT_TOKEN)

@bolt_app.event("message")
def handle_message_events(body, say):
    event = body.get("event", {})
    text = event.get("text", "")
    channel = event.get("channel", "")

    try:
        logging.debug(f"Received message text: {text}")
        features = eval(text)
        logging.debug(f"Received features: {features}")
        
        if all(key in features for key in ["Customer", "Source", "SourceType", "FailType", "FailSummary"]):
            response = requests.post("http://127.0.0.1:5000/predict", json={"features": features})
            logging.debug(f"Data sent to ML model: {features}")
            logging.debug(f"Response from ML model: {response.status_code}, {response.text}")

            if response.status_code == 200:
                solution = response.json().get("Solution", "No solution found.")
                response_text = f"Solution: {solution}"
                say(text=response_text, channel=channel)
            else:
                error_message = f"Could not get a response from the model. HTTP Status Code: {response.status_code}, Response Text: {response.json().get('error', 'Unknown error')}"
                say(text=error_message, channel=channel)
                logging.error(error_message)
        else:
            error_message = "The message lacks required 'features' keys."
            say(text=error_message, channel=channel)
            logging.error(error_message)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        say(text=error_message, channel=channel)
        logging.error(error_message)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features")
    
    required_features = ["Customer", "Source", "SourceType", "FailType", "FailSummary"]
    missing_features = [feature for feature in required_features if feature not in features]

    if missing_features:
        error_message = f"Missing features: {missing_features}"
        logging.error(error_message)
        return jsonify({
            "error": error_message,
            "missing_features": missing_features
        }), 400

    try:
        input_df = pd.DataFrame([features])
        logging.debug(f"Input DataFrame for prediction: {input_df}")

        
        prediction = pipelineMaps.predict(input_df)
        logging.debug(f"Model prediction: {prediction}")

        solution = label_encoder.inverse_transform(prediction)[0]
        logging.debug(f"Decoded solution: {solution}")

        return jsonify({"Solution": solution})
    except Exception as e:
        error_message = f"Prediction error: {e}"
        logging.error(error_message)
        return jsonify({
            "error": error_message
        }), 500

if __name__ == '__main__':
    handler = SocketModeHandler(bolt_app, APP_TOKEN)
    handler.start()
    app.run(port=5000, debug=True)
