from flask import Flask, render_template, request, jsonify
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
import os

app = Flask(__name__)

# Load model and tokenizer during application startup (TensorFlow)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = TFAutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Initialize chat history outside the function
chat_history_ids = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):
    global chat_history_ids

    # Let's chat for 1 line to reduce memory usage.
    for step in range(1):
        # encode the new user input, add the eos_token and return a tensor in TensorFlow
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='tf')

        # append the new user input tokens to the chat history
        if chat_history_ids is not None:
            bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids], axis=-1)
        else:
            bot_input_ids = new_user_input_ids

        # generate a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last output tokens from bot
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

