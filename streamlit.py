import streamlit as st
from streamlit_chat import message
import requests
import json


# Constants 
# HOST = 'http://127.0.0.1:8080/api'  # HOST SERVER URL
HOST = 'http://127.0.0.1:8080'  # HOST SERVER URL
# HOST = 'https://chat-bot-backend-an9j.onrender.com/api'
####### ENDPOINT of different Api ####################333
ENDPOINT1 = '/api/predict'  
# ENDPOINT2 = '/train'
# ENDPOINT3 = '/addQuestion'

url ='https://www.greenlife.co.ke/wp-content/uploads/2022/04/Fall-Armyworm-Greenlife-1566x783-1.jpg'

st.image(url, width=750)
st.title("Crop Diseases Chat-Bot")

isYes = False
index = None
question = ""


## store session_state variable to load after old session after clicking buttons or show different messages 
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
    """
        get user input in a text box and store in input_text variable
    """
    input_text = st.text_input("Question: ","", key=input)
    return input_text
    

## store user input
user_input = get_text() 
# _, btn1, btn2 = st.columns([4,0.5,0.5])
isSend = st.button("Send")
# isTrain = btn2.button("Train")

## check if send button is click or not
if isSend:
    st.session_state.past.append(user_input)
    data = {'question':user_input}
    ## hit the predict api and get the response
    res = requests.post(HOST+ENDPOINT1,json=data)
    data = json.loads(res.text) 
    index = data['data'][1]
    question = data['data'][2]
    st.session_state.ind = index
    st.session_state.question = question
    st.session_state.generated.append(data['data'][0])

### show all message that user ask and the response that the user get
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')