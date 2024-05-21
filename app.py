import logging
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
APP_TOKEN = os.getenv("APP_TOKEN")
ML_MODEL_URL = "http://127.0.0.1:5000/predict"

bolt_app = App(token=SLACK_BOT_TOKEN)

@bolt_app.event("message")
def handle_message_events(body, say):
    event = body.get("event", {})
    text = event.get("text", "")
    channel = event.get("channel", "")

    try:

        logging.debug(f"Gelen mesaj metni: {text}")
        features = eval(text)
        logging.debug(f"Gelen özellikler: {features}")
        
        if 'features' in features:
            response = requests.post(ML_MODEL_URL, json={"features": features})
            logging.debug(f"ML Modeline gönderilen veri: {features}")
            logging.debug(f"ML Modelinden gelen yanıt: {response.status_code}, {response.text}")

            if response.status_code == 200:
                solution = response.json().get("Solution", "Çözüm bulunamadı.")
                response_text = f"Solution: {solution}"
                say(text=response_text, channel=channel)
            else:
                error_message = f"Modelden yanıt alınamadı. HTTP Status Code: {response.status_code}, Response Text: {response.json().get('error', 'Bilinmeyen hata')}"
                say(text=error_message, channel=channel)
                logging.error(error_message)
        else:
            error_message = "Gelen mesajda 'features' anahtarı bulunamadı."
            say(text=error_message, channel=channel)
            logging.error(error_message)
    except Exception as e:
        error_message = f"Bir hata oluştu: {e}"
        say(text=error_message, channel=channel)
        logging.error(error_message)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features")
    
    required_features = ["Customer", "Source", "SourceType", "FailType", "FailSummary"]
    missing_features = [feature for feature in required_features if feature not in features]

    if missing_features:
        error_message = f"Eksik özellikler var: {missing_features}"
        logging.error(error_message)
        return jsonify({
            "error": error_message,
            "missing_features": missing_features
        }), 400

    solution = {
        "Solution": "temp tabloda product desc kolon uzunluğu artırıldı"
    }

    return jsonify(solution)

if __name__ == '__main__':
    handler = SocketModeHandler(bolt_app, APP_TOKEN)
    handler.start()
    app.run(port=5000, debug=True)
