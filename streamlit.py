import nltk
import numpy as np
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from streamlit_chat import message

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

## EDIT DISTANCE FUNCTION To Correct the misspelled words
def minDis(s1, s2, n, m, dp):
    if n == 0:
        return m
    if m == 0:
        return n
    if dp[n][m] != -1:
        return dp[n][m]
    if s1[n - 1] == s2[m - 1]:
        if dp[n - 1][m - 1] == -1:
            dp[n][m] = minDis(s1, s2, n - 1, m - 1, dp)
            return dp[n][m]
        else:
            dp[n][m] = dp[n - 1][m - 1]
            return dp[n][m]
    else:
        if dp[n - 1][m] != -1:
            m1 = dp[n - 1][m]
        else:
            m1 = minDis(s1, s2, n - 1, m, dp)
        if dp[n][m - 1] != -1:
            m2 = dp[n][m - 1]
        else:
            m2 = minDis(s1, s2, n, m - 1, dp)
        if dp[n - 1][m - 1] != -1:
            m3 = dp[n - 1][m - 1]
        else:
            m3 = minDis(s1, s2, n - 1, m - 1, dp)
        dp[n][m] = 1 + min(m1, min(m2, m3))
        return dp[n][m]

# Streamlit app
url = 'https://www.greenlife.co.ke/wp-content/uploads/2022/04/Fall-Armyworm-Greenlife-1566x783-1.jpg'
st.image(url, width=750)
st.title("Crop Diseases Chat-Bot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'index' not in st.session_state:
    st.session_state.index = []

if 'ind' not in st.session_state:
    st.session_state.ind = -1

if 'question' not in st.session_state:
    st.session_state.question = ""

def get_text():
    input_text = st.text_input("Question: ", "", key="input")
    return input_text

user_input = get_text()
isSend = st.button("Send")

if isSend:
    st.session_state.past.append(user_input)
    msg = main_(user_input.lower())
    index = msg[1]
    question = msg[2]
    st.session_state.ind = index
    st.session_state.question = question
    st.session_state.generated.append(msg[0])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
