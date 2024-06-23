#!/usr/bin/env python
# coding: utf-8


# #### Libraries that we use for giving the response

import nltk
import numpy as np
import json
import pickle
import random 
from nltk.stem import WordNetLemmatizer 
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import tensorflow as tf

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('crops.json') as json_file:
    intents = json.load(json_file)

# Load pre-trained model and supporting files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    return embeddings

def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    ERROR_THRESHOLD = 0.20
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if tag in i['tags']:
            return random.choice(i['responses']), intents_json['intents'].index(i), tag
    return "I do not know about it", -1, ""

def main_(message: str):
    ints = predict_class(message)
    if ints:
        return get_response(ints, intents)
    return "I do not know about it", -1, ""

# # In[ ]:
