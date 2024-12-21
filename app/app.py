import streamlit as st
import requests
import tensorflow as tf
import base64
import nltk
import re
import os
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
load_dotenv()

st.title("Emotion Text Analyze")

end_point = os.environ.get("API_PREDICTION")

stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def clean_text(text):
    """
        Membersihkan teks input pengguna untuk mempersiapkan proses analisis emosi.

        Args:
            text (str): Teks input dari pengguna.

        Returns:
            str: Teks yang telah dibersihkan, diturunkan menjadi huruf kecil,
                 dihapus dari URL, tanda baca, angka, dan kata-kata umum (stopwords).
    """
    text_clean = text.lower()
    text_clean = re.sub(r'http\S+', '', text_clean)
    text_clean = re.sub(r'www\.\S+', '', text_clean)
    text_clean = re.sub(r'[^\w\s]', '', text_clean)
    text_clean = re.sub('\w*\d\w*', '', text_clean)
    tokens = word_tokenize(text_clean)
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

def _bytes_feature(value):
    """
        Membuat fitur byte list yang kompatibel dengan TensorFlow.

        Args:
            value (bytes): Data teks yang dikodekan ke dalam format byte.

        Returns:
            tf.train.Feature: Fitur TensorFlow dalam bentuk byte list.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_text(text):
    """
        Mengonversi teks yang telah dibersihkan menjadi format serialized TensorFlow Example.

        Args:
            text (bytes): Teks yang telah dibersihkan dan dikodekan dalam format byte.

        Returns:
            bytes: Data serialized TensorFlow Example dalam bentuk byte.
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'text': _bytes_feature(text)
        }))
    serialized_example = example.SerializeToString()
    return serialized_example

def generate_response(input_text):
    """
        Membersihkan teks input, mengirimkannya ke model TensorFlow Serving,
        dan menampilkan hasil prediksi emosi kepada pengguna.

        Args:
            input_text (str): Teks input dari pengguna.

        Displays:
            str: Label emosi yang diprediksi oleh model, ditampilkan di antarmuka Streamlit.
    """
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
    """
        Membuat formulir input Streamlit untuk menerima teks dari pengguna
        dan memproses analisis emosi berdasarkan input tersebut.
    """
    text = st.text_area("Enter text:")
    submitted = st.form_submit_button("Submit")
    if not submitted:
        st.info("silahkan masukkan kalimat")
    else:
        generate_response(text)