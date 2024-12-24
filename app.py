
from flask import Flask, render_template, request, jsonify
import requests
import os
from model import ask_weaviate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_APIKEY")

def generate_openai_response(message):
    if "SNHU TEXT" in message:
        # Strip the keyword from the message before sending it to Weaviate
        clean_message = message.replace("SNHU TEXT", "").strip()
        return ask_weaviate(clean_message)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
        "max_tokens": 4000,
        "temperature": 0.8,
        "model": "gpt-3.5-turbo"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    result = response.json()
    return result["choices"][0]["message"]["content"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    assistant_reply = generate_openai_response(user_message)
    return jsonify({'response': assistant_reply})

if __name__ == '__main__':
    app.run()



