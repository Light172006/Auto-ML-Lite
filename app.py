import streamlit as st
import pandas as pd
import numpy as np
import classifation

st.set_page_config(page_title = 'Auto ML Lite')     # setting the page configuration

st.header("Auto ML Lite")

file = st.file_uploader('Enter Your Dataset',type='csv')

if 'model' not in st.session_state:
    st.session_state.model = ''

st.selectbox("Enter model Type: ",options=['Classification','Regression'],key = 'model')  # indentifing which type of model it is

if 'target' not in st.session_state:
    st.session_state.target = ''

@st.cache_resource
def input_file(x):
    data = pd.read_csv(x)            #getting the csv file
    data = pd.DataFrame(data)           #converting it into DataFrame
    return data

if file is not None:                        
    data = input_file(file)
    features_names = data.columns           #extracting feature name

if st.text_input('Enter The Target',key = 'target'):
    st.write(st.session_state.target)
    if st.session_state.target not in features_names:
        st.error('Wrong input!, Enter Again')
    else:
        st.success('Target Succesfully Entered')
        data = classifation.clean(data,st.session_state.target)
        X_axis = data.drop([st.session_state.target],axis=1)
        y_axis = data[st.session_state.target]
        X_axis = classifation.feature_engineering(X_axis)
        X_axis = classifation.X_encoding(X_axis)
        X_axis = classifation.dimention_reduction(X_axis)
        y_axis = classifation.y_encoding(y_axis)


if st.button('Train'):
    st.warning('Your Model Is Training')
    st.spinner("Training model...")
    best_model,best_score,  = classifation.train(X_axis,y_axis)
    st.chat_message('Traing Succesfull')
    st.write(f'Best Model: {best_model} ')
    st.write(f'Score: {round(best_score,4)*100}')

