import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding,SimpleRNN
from tensorflow.keras.models import load_model
import streamlit as st


# decoding the integers into the words
word_index=imdb.get_word_index()
reverse_word_index={value:key for(key,value)in word_index.items()}

#save the model in h5 format

model1=load_model('Imdb_review_model.h5')
def decode_review(encoded_review):
     return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

from tensorflow.keras.preprocessing.sequence import pad_sequences # Import the specific pad_sequences
def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[(word_index.get(word,2) +3) for word in words]
  padded_review=pad_sequences([encoded_review],maxlen=max_len)
  return padded_review

def predict_sentiment(review):
  preprocesstext=preprocess_text(review)
  prediction=model1.predict(preprocesstext)
  sentiment ="positive" if prediction[0][0]>0.5  else "Negative"
  return sentiment, prediction[0][0]


st.title("Movie Review Sentiment Analysis")
st.write("enter the movie review to classify whether it is positive or negative")

Review_as_input= st.text_area('Movie Review')
if st.button('classify'):
   preprocesstext=preprocess_text(Review_as_input)
   
   prediction=load_model.predict(preprocesstext)   
   sentiment ="positive" if prediction[0][0]>0.5  else "Negative"
   st.write(f'sentiment,{sentiment}')
   st.write(f'prediction score:,{prediction[0][0]}')
else:
     st.write(f'please enter the movie review')   