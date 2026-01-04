import streamlit as st
import pandas as pd
import numpy as np
import training
import preprocessing
import joblib

st.set_page_config(page_title = 'Auto ML Lite')     # setting the page configuration

st.header("Auto ML Lite")

file = st.file_uploader('Enter Your Dataset',type='csv')

if 'model' not in st.session_state:             # for storing model type
    st.session_state.model = ''

st.selectbox("Enter model Type: ",options=['Classification','Regression'],key = 'model')  # indentifing which type of model it is

if 'target' not in st.session_state:        # for storing target name
    st.session_state.target = ''

@st.cache_resource
def input_file(x):
    data = pd.read_csv(x)            #getting the csv file
    data = pd.DataFrame(data)           #converting it into DataFrame
    return data

if file is not None:                        
    data = input_file(file)
    features_names = data.columns           #extracting feature name

if st.text_input('Enter The Target',key = 'target'):        #taking input of target 
    if st.session_state.target not in features_names:
        st.error('Wrong input!, Enter Again')
    else:
        st.success('Target Succesfully Entered')
        data = preprocessing.clean(data,st.session_state.target)           #cleaning the dataset
        X_axis = data.drop([st.session_state.target],axis=1)                
        y_axis = data[st.session_state.target]
        X_axis = preprocessing.feature_engineering(X_axis)                 #feature engineering
        X_axis = preprocessing.X_encoding(X_axis)                          #one hot encoding x axis
        X_axis = preprocessing.data_scale(X_axis)
        y_axis = preprocessing.y_encoding(y_axis)                          #encoding y axis
            
        if st.button('Train'):                                              #traing the model
            st.warning("Training model...")
            if st.session_state.model == 'Classification':
                best_model_name,best_score,best_model  = training.c_train(X_axis,y_axis)
            else:
                best_model_name,best_score,best_model  = training.r_train(X_axis,y_axis)
            st.success('Traing Succesfull')
            st.write(f'Best Model: {best_model_name}')
            st.write(f'Score: {round(best_score,4)*100}')
            joblib.dump(best_model,'best_model.pkl')                         # saving the best model
            with open('best_model.pkl','rb') as f:
                st.write('### Here is your trained model:')
                st.download_button("Best Model",data=f,file_name='best_model.pkl')