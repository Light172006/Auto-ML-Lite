import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


imputer = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
scale = StandardScaler()
pca = PCA(0.95).set_output(transform='pandas')

def clean(x,target):               
    x = x.dropna(axis = 1,how = 'all')              #droping column with all null values
    x = x.dropna(subset = [target])                 #droping all na values from target 
    imputed_data = imputer.fit_transform(x)
    #print(imputed_data.dtype.names)
    return pd.DataFrame(imputed_data,columns= x.columns)

def feature_engineering(x):
    x = x.apply(pd.to_numeric, errors='ignore')         #converting all features to numeric
    cat_features = x.select_dtypes(include='object')    #selecting catagorical features
    num_features = x.select_dtypes(exclude='object')    #selecting numeric features
    print(cat_features.columns)
    for col in cat_features.columns:                    #droping columns with more than one unique values
        if len(cat_features[col].unique()) > 5:
            x = x.drop(col,axis = 1)
            print(col)

    for col in num_features.columns:                    #droping columns with high varience
        varience = np.var(num_features[col])
        print(f'{col} : {varience}')
        if varience > -0.5 and varience < .5:
            x = x.drop(col,axis = 1)
            print(col)

    return x

def X_encoding(x):
    cat_features = x.select_dtypes(include='object')            #selecting catagorical features
    encoded = ohe.fit_transform(cat_features)
    result = x.drop(cat_features,axis = 1)
    result = pd.concat([result,encoded],axis=1)
    return result

def y_encoding(y):
    try:
        y = y.apply(pd.to_numeric)              #converting target to numeric
        print('done',y.iloc[0].dtype)
        return y
    except:
        uni = y.iloc[0].unique()
        num = len(uni)
        for i in range(num):
            y.iloc[0] = y.iloc[0].apply(lambda x: i if x == uni[i] else x)
        return y

def data_scale(x):
    x = x.apply(pd.to_numeric, errors='ignore')             #converting all features to numeric
    numeric_col = x.select_dtypes(include='number')
    numeric_feature_name = numeric_col.columns
    x[numeric_feature_name] = scale.fit_transform(x[numeric_feature_name])
    return x

def dimention_reduction(x):
    x = pca.fit_transform(x)
    return x
