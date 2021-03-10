import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



st.write("# Fake Message Recognition Engine")

message_text = st.text_area("Enter a message for  evaluation")

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

model = joblib.load('spam_classifier.joblib')

message_submit = st.button('Evaluate')

if message_submit:

    label = (model.predict([message_text])[0])
    spam_prob = (model.predict_proba([message_text]))
    if label=="spam":
        label="fake"
    elif label =="ham":
        label="real"

    result = {'label': label, 'probability': spam_prob[0][1]}

    st.write(result)


explain_pred = st.button('Explain Predictions')

if explain_pred:
	with st.spinner('Generating explanations'):
		class_names = ['Real', 'Fake']
		explainer = LimeTextExplainer(class_names=class_names)
		exp = explainer.explain_instance(message_text,model.predict_proba, num_features=10)
		components.html(exp.as_html(), height=800)
