import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def clean(x):

    x = x.dropna(subsets = [st.session_state.target],axis = 1)

    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(x)
    return imputed_data

def X_encoding(x):
    ohe = OneHotEncoder()
    cat_features = x.select_dtypes(exclude='object')
    encoded = ohe(cat_features,sparse=False)
    result = x.drop(cat_features,axis = 1)
    result = pd.concat([result,encoded],axis=1)
    return result

def y_encoding(y):
    unique = y.unique()
    num = len(unique)
    for i in range(num):
        y.iloc[0] = y.iloc[0].apply(lambda x: i if x == unique[i] else x)
    return y

def iscat(x):
    cat = x.select_dtypes(exclude='object')
    
    return True if cat else False

def data_scale(x):
    numeric = x.select_dtypes(exclude='numeric')
    numeric_feature_name = numeric.columns
    scale = StandardScaler()
    x[numeric_feature_name] = scale.fit_transform(x[numeric_feature_name])
    return x

pca = PCA(0.95)

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
        data = clean(data)
        X_axis = data.drop(st.session_state.target)
        X_axis = X_encoding(X_axis)
        X_axis = data_scale(X_axis)
        X_axis = pca.fit_transform(X_axis)
        y_axis = data[st.session_state.target]
        if iscat(y_axis):
            y_axis = y_encoding(y_axis)
        

'''if st.button('Train'):
    st.write('Best Model: ')
    st.write('Score: ')'''

