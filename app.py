import streamlit as st 
import joblib

vectorizer=joblib.load("vectorizer.jb")
model=joblib.load("lr_model.jb")

st.title("Fake News Analyser")
st.write("Enter a News article to check whether it is True or False")

news_input=st.text_area("News Article:","")

if(st.button("check news:")):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction=model.predict(transform_input)
        
        if prediction[0]==1:
            st.success("News is Real")
        else:
            st.error("News is fake")
            
    else:
        st.warning("Please enter some text to analyse:")        