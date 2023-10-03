import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download("punkt")
nltk.download("stopwords")

def textTransformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email / SMS spam classifier')
input_sms = st.text_input('Enter your message')

# Preprocess
transformed_sms = textTransformer(input_sms)

# Vectorize
vector_input = tfidf.transform([transformed_sms])

# Predict
result = model.predict(vector_input)[0]

# Display
if st.button('Classify'):
    if result == 1:
        st.header("Spam")
    else:
        st.header('Not spam')
