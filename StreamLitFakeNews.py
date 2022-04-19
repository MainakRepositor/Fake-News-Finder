import streamlit as st
import pandas as pd

import spacy
import time
from PIL import Image

from io import StringIO 

def load_model():
    #declare global variables
    global nlp
    global textcat        
    nlp = spacy.load('models/fakenews')
    textcat = nlp.get_pipe('textcat')


def predict(news):
    print("news = ", news)
    news = [news]
    txt_docs = list(nlp.pipe(news))
    scores, _ = textcat.predict(txt_docs)
    print(scores)
    predicted_classes = scores.argmax(axis=1)
    print(predicted_classes)
    result = ['real' if lbl == 0 else 'fake' for lbl in predicted_classes]  
    print(result)
    return(result)


def run():
    st.sidebar.info('You can either enter the news item online in the textbox or upload a txt file')    
    st.set_option('deprecation.showfileUploaderEncoding', False)       
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Txt file"))    
    image = Image.open('data/20170715_162310.jpg')
    
    
    st.title("Predicting fake news")
    st.header('This app is created to predict if a news item is real or fake')
 
    if add_selectbox == "Online":
        text1 = st.text_area('Enter news text')
        output = ""
        if st.button("Predict"):
            output = predict(text1)
            output = str(output[0])  # since its a list, get the 1st item
            st.success(f"The news item is {output}")     
            st.balloons()   
    elif add_selectbox == "Txt file":        
        output = ""
        file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])           
        if st.button("Predict"):
            text_news = file_buffer.read()  
            
            # in the latest stream-lit version ie. 68, we need to explicitly convert bytes to text
            st_version = st.__version__  # eg 0.67.0
            versions = st_version.split('.')           
            if int(versions[1]) > 67:
                text_news = text_news.decode('utf-8')
            
            print(text_news)
            output = predict(text_news)
            output = str(output[0])
            st.success(f"The news item is {output}")      
            st.balloons()    
              
    val_path = "data/val.csv"     
    df = st.cache(pd.read_csv)(val_path)    
    is_check = st.checkbox("Display validation data")
    if is_check:
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        st.write(df)

    

    
    
if __name__ == "__main__":
    load_model()
    run()