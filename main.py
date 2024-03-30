import streamlit as st
import requests
import tensorflow as tf
import base64
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pprint import PrettyPrinter
nltk.download('stopwords')
nltk.download('punkt')

st.title("Emotion Text Classification")

end_point = 'http://<external-ip>:8501/v1/models/1709760128:predict'


stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_text(text):
    example = tf.train.Example(features=tf.train.Features(feature={
        'text': _bytes_feature(text)
        }))
    serialized_example = example.SerializeToString()
    return serialized_example



def generate_response(input_text):
    text = clean_text(input_text)
    text = text.encode('utf-8')
    example = serialize_text(text)
    json_data = {
        "signature_name":"serving_default",
        "instances":[
            {
                "examples":{"b64": base64.b64encode(example).decode('utf-8')}
            }
        ]
    }
    response = requests.post(end_point, json=json_data)
    prediction = tf.argmax(response.json()["predictions"][0]).numpy()
    map_labels = {0: "sadness",
                  1: "joy",
                  2: "love",
                  3: "anger",
                  4: "fear",
                  5: "surprise"}
    
    st.info(map_labels[prediction])


with st.form("my_form"):
    text = st.text_area("Enter text:")
    submitted = st.form_submit_button("Submit")
    generate_response(text)