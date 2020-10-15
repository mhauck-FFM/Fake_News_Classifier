import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests

from helpers.tokenize_text import tokenize

categories = ['True', 'Satire/Parody', 'Misleading Content', 'Manipulated Content', 'False Connection', 'Imposter Content']

@st.cache(allow_output_mutation = True, show_spinner=False)
def load_fake_news_model():
    return load_model('./model/reddit_classifier.h5')

@st.cache(allow_output_mutation = True, show_spinner=False)
def load_tokenizer():
    return pickle.load(open('./helpers/tokenizer_reddit.pkl', 'rb'))

@st.cache(allow_output_mutation = True, show_spinner=False)
def get_title(url_list, url_class = 'h1', user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36'):
    headers ={'User-Agent': user_agent}
    with requests.Session() as s:
        r = s.get(url_list, headers = headers)
        soup = BeautifulSoup(r.content, 'lxml')
        return str(soup.select(url_class, class_ = '_eYtD2XCVieq6emjKBH3m')[0].get_text())

if __name__ == '__main__':

    st.write("""
    # Reddit Post Classifier

    """)

    text_default = 'Reddit is awesome'
    text_input = st.text_input('Enter title or direct Reddit post link here:', text_default)

    if text_input == '':
        text_input = get_title('https://www.reddit.com/r/worldnews/', url_class = 'h3')

    if text_input[0:5] == 'https' or text_input[0:4] == 'http' or text_input[0:4] == 'www':
        text_input = get_title(text_input, url_class = 'h1')

    tkz = load_tokenizer()
    fake_news_model = load_fake_news_model()

    pred = fake_news_model.predict(tokenize(text_input, 25, tkz))

    pred = np.round(pred[0]*100., decimals = 0)

    pred_df = pd.DataFrame({'category': ['True', 'Satire/Parody', 'Misleading Content', 'Manipulated Content', 'False Connection', 'Imposter Content'], 'probability': pred})

    if text_input != text_default:
        st.write('')
        st.write("""
            Your Reddit post
        """)
        st.write("""
            "*{}*"
        """.format(text_input))
        st.write("""
            falls with **{}** % certainty into the category **{}**!
        """.format(np.max(pred), categories[np.argmax(pred)]))
        st.write('')


        fig, ax = plt.subplots()
        plt.bar(pred_df['category'], pred_df['probability'])
        plt.xticks(rotation='vertical')
        plt.ylabel('Probability / %')
        st.pyplot(fig)

    else:
        st.write('That is **100 %** true!')
