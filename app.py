import streamlit as st
import pandas as pd
import numpy as np
import regression

st.set_page_config(page_title = 'Auto ML Lite')     # setting the page configuration

st.header("Auto ML Lite")

file = st.file_uploader('Enter Your Dataset',type='csv')

if 'model' not in st.session_state:
    st.session_state.model = list()

st.selectbox("Enter model Type: ",options=['Classification','Regression'],key = 'model')  # indentifing which type of model it is

if 'target' not in st.session_state:
    st.session_state.target = ''

''' if 'X_axis' not in st.session_state:
    st.session_state.X_axis = pd.DataFrame(np.zeros(8).reshape(2,4)) '''

''' if 'y_axis' not in st.session_state:
    st.session_state.y_axis = pd.DataFrame(np.zeros(8).reshape(2,4))'''

if file is not None:                        
    data = pd.read_csv(file)            #getting the csv file
    data = pd.DataFrame(data)           #converting it into DataFrame

features_names = data.columns           #extracting feature name

if st.text_input('Enter The Target',key = 'target'):
    st.write(st.session_state.target)
    if st.session_state.target not in features_names:
        st.error('Wrong input!, Enter Again')
    else:
        st.success('Target Succesfully Entered')
        
        

'''if st.button('Train'):
    st.write('Best Model: ')
    st.write('Score: ')'''

