import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras import layers as Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

def train(model, inputs, outputs, epochs=1000, verb=1):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(inputs, outputs, epochs=epochs, verbose=verb)
    return model


def load_model():
    model = keras.models.load_model('chat_model')
    return model


def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_label_encoder():
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return lbl_encoder


def load_intents():
    with open('intents.json') as file:
        data = json.load(file)
    return data


def get_tag(inp, model, tokenizer, lbl_encoder, max_len):
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                         truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    return tag[0]


def get_response(tag, data):
    for i in data['intents']:
        if i['tag'] == tag:
            response = np.random.choice(i['responses'])
            return response


def ask_feedback(response):
    print(Fore.LIGHTBLUE_EX + "Did I answer your question? (yes/no): " + Style.RESET_ALL, end="")
    feedback = input()
    if feedback.lower() == "yes":
        return True
    elif feedback.lower() == "no":
        return False


def update_intents(tag, inp, correct_response, data):
    for i in data['intents']:
        if i['tag'] == tag:
            i['responses'].append(correct_response)
            break
    else:
        new_intent = {
            "tag": "tag_" + str(len(data["intents"]) + 1),
            "patterns": [inp],
            "responses": [correct_response]
        }
        data["intents"].append(new_intent)
    with open('intents.json', 'w') as file:
        json.dump(data, file, indent=4)


def retrain_model(data):
    training_sentences = []
    training_labels = []
    labels = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)                    
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', maxlen=max_len)
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    model.save("chat_model")

    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # to save the fitted label encoder
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


# load the pre-trained model and tokenizer
model = load_model()
tokenizer = load_tokenizer()
lbl_encoder = load_label_encoder()
max_len = 20

print(Fore.YELLOW + "Bot is ready to answer your questions! (type 'quit' to exit)" + Style.RESET_ALL)
while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break

    tag = get_tag(inp, model, tokenizer, lbl_encoder, max_len)
    response = get_response(tag, data)
    print(Fore.GREEN + "Bot: " + Style.RESET_ALL + response)

    if not ask_feedback(response):
        print(Fore.LIGHTBLUE_EX + "What should I have said?" + Style.RESET_ALL, end="")
        correct_response = input()
        update_intents(tag, inp, correct_response, data)
        retrain_model(data)
        model = load_model()
        tokenizer = load_tokenizer()
        lbl_encoder = load_label_encoder()
