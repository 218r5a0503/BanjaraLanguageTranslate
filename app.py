from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import pickle

app = Flask(__name__)

# Load model and metadata
model = load_model("s2s_model.keras")
data = pickle.load(open("training_data.pkl", "rb"))

input_characters = data['input_characters']
target_characters = data['target_characters']
max_input_length = data['max_input_length']
max_target_length = data['max_target_length']
num_en_chars = data['num_en_chars']
num_dec_chars = data['num_dec_chars']

# Find LSTM and Dense layers
encoder_lstm = None
decoder_lstm = None
decoder_dense = None

for layer in model.layers:
    if isinstance(layer, LSTM):
        if encoder_lstm is None:
            encoder_lstm = layer
        else:
            decoder_lstm = layer
    elif isinstance(layer, Dense):
        decoder_dense = layer

# Build encoder
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# Build decoder
decoder_inputs = model.input[1]
decoder_state_input_h = Input(shape=(256,), name="input_h")
decoder_state_input_c = Input(shape=(256,), name="input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_states = [state_h_dec, state_c_dec]

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Encode input sequence into one-hot
def encode_input_sequence(input_seq):
    encoder_input_data = np.zeros((1, max_input_length, num_en_chars), dtype='float32')
    for t, char in enumerate(input_seq.lower()):
        if char in input_characters:
            encoder_input_data[0, t, input_characters.index(char)] = 1.
    return encoder_input_data

# Decode using encoder & decoder models
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_dec_chars))
    target_seq[0, 0, target_characters.index('\t')] = 1.

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_characters[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_target_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_dec_chars))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    english_text = request.form.get("english_text")
    input_seq = encode_input_sequence(english_text)
    banjara_text = decode_sequence(input_seq)
    return jsonify({"banjara": banjara_text})

if __name__ == "__main__":
    app.run(debug=True)
