from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from SwornOfficerChatbot import ContentChatbot
import os


app = Flask(__name__)
CORS(app)
chatbot = ContentChatbot()


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message')

        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        response = chatbot.chat(user_input)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return send_file('templates/index.html')


if __name__ == '__main__':
    app.run()
