import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB , GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def clean(x,target):
    x = x.dropna(axis = 1,how = 'all')
    x = x.dropna(subset = [target])
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_data = imputer.fit_transform(x)
    #print(imputed_data.dtype.names)
    return pd.DataFrame(imputed_data,columns= x.columns)

def feature_engineering(x):
    x = x.apply(pd.to_numeric, errors='ignore')
    cat_features = x.select_dtypes(include='object')
    num_features = x.select_dtypes(exclude='object')
    print(cat_features.columns)
    for col in cat_features.columns:
        if len(cat_features[col].unique()) > 5:
            x = x.drop(col,axis = 1)
            print(col)

    for col in num_features.columns:
        varience = np.var(num_features[col])
        print(f'{col} : {varience}')
        if varience > -0.5 and varience < .5:
            x = x.drop(col,axis = 1)
            print(col)

    return x

def X_encoding(x):
    ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
    cat_features = x.select_dtypes(include='object')
    encoded = ohe.fit_transform(cat_features)
    result = x.drop(cat_features,axis = 1)
    result = pd.concat([result,encoded],axis=1)
    return result

def y_encoding(y):
    try:
        y.iloc[0] = pd.to_numeric(y.iloc[0])
        return y
    except:
        uni = y.iloc[0].unique()
        num = len(uni)
        for i in range(num):
            y.iloc[0] = y.iloc[0].apply(lambda x: i if x == uni[i] else x)
        return y

def iscat(x):
    return True if x.iloc[0].dtype == 'object'  else False

def data_scale(x):
    x = x.apply(pd.to_numeric, errors='ignore')
    numeric_col = x.select_dtypes(include='number')
    numeric_feature_name = numeric_col.columns
    scale = StandardScaler()
    x[numeric_feature_name] = scale.fit_transform(x[numeric_feature_name])
    return x

def dimention_reduction(x):
    pca = PCA(0.95).set_output(transform='pandas')
    x = pca.fit_transform(x)
    return x

classification_models = {
                           "LogisticRegression" : {
                                                        'model' : LogisticRegression(),
                                                        'para' : {
                                                                      'penalty': ['l1','l2'],
                                                                      'C' : [0.001,0.1,1,5,10]
                                                        }
                           },
                            "RandomForestClassifier" : {
                                                        'model' : RandomForestClassifier(),
                                                        'para' : {
                                                                'n_estimators': [200,300],
                                                                'max_depth': [10, 20, None],
                                                                'min_samples_leaf': [1, 5, 10],
                                                                'max_features': ['sqrt', 'log2']
                                                                }
                             },
                             "KNN" : {
                                        'model' : KNeighborsClassifier(),
                                        'para' : {
                                                    'n_neighbors' : range(1,11)
                                                 }
                            },
                             "SVC" : {
                                        'model' : SVC(),
                                        'para' : {
                                                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                                    'C' : [0.001,0.1,1,5,10],
                                                    'gamma' : ['scale', 'auto']
                                                 }
                            },
                            "BernoulliNB" : {
                                        'model' : BernoulliNB(),
                                        'para' : {
                                                    'alpha': [0.01,0.1,10]
                                                 }
                            },
                            "GaussianNB" : {
                                        'model' : GaussianNB(),
                                        'para' : {
                                                    'var_smoothing' : [1e-8,1e-9,1e-10,1e-11]
                                                 }
                            },
}

def train(x,y):
    score = 0
    m_name = ''
    paramiter = []
    for model_name,model in classification_models.items():
        grid = GridSearchCV(model['model'],model['para'],cv=3,n_jobs=-1,scoring='r2')
        grid.fit(x,y.values.ravel())
        if grid.best_score_ > score:
            score = grid.best_score_
            m_name = model_name
            paramiter = grid.best_params_
    return m_name , paramiter , score