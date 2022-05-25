
import streamlit as st
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def load_model(MODEL):
	tokenizer = AutoTokenizer.from_pretrained(MODEL)
	config = AutoConfig.from_pretrained(MODEL)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL)
	model.save_pretrained(MODEL)
	tokenizer.save_pretrained(MODEL)
	return model, tokenizer

def prediction(text, model, tokenizer):
	text = preprocess(text)
	encoded_input = tokenizer(text, return_tensors='pt')
	output = model(**encoded_input)
	scores = output[0][0].detach().numpy()
	scores = softmax(scores)
	lidx = np.argmax(scores)
	return lidx, np.round(float(np.max(scores)), 3)

SENTIMENT_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Postive']

HATE_MODEL = f"cardiffnlp/twitter-roberta-base-hate"
HATE_LABELS = ['Not-Hate', 'Hate']

h_model, h_tokenizer = load_model(HATE_MODEL)
#s_model, s_tokenizer = load_model(SENTIMENT_MODEL)

#print(prediction('Fuck you immigrants', h_model, h_tokenizer))
#print(prediction('Fuck you immigrants', s_model, s_tokenizer))

st.title("Social media study of Toxic Speech towards Migrants!")
tweet = st.text_input('Input Tweet', '')
if st.button('Analyse'):
	st.write(f'Results for the tweet: {tweet}')
	lidx, prob = prediction(tweet, h_model, h_tokenizer)
	st.write(f'The Tweet is flagged as {HATE_LABELS[lidx]} with a confidence score {prob}')

	#lidx, prob = prediction(tweet, s_model, s_tokenizer)
	#st.write(f'The Tweet is flagged as {SENTIMENT_LABELS[lidx]} sentiment with a confidence score {prob}')

	

