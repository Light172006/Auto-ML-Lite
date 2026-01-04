import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB , GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

imputer = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
scale = StandardScaler()
pca = PCA(0.95).set_output(transform='pandas')

def clean(x,target):
    x = x.dropna(axis = 1,how = 'all')
    x = x.dropna(subset = [target])
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
    cat_features = x.select_dtypes(include='object')
    encoded = ohe.fit_transform(cat_features)
    result = x.drop(cat_features,axis = 1)
    result = pd.concat([result,encoded],axis=1)
    return result

def y_encoding(y):
    try:
        y = y.apply(pd.to_numeric)
        print('done',y.iloc[0].dtype)
        return y
    except:
        uni = y.iloc[0].unique()
        num = len(uni)
        for i in range(num):
            y.iloc[0] = y.iloc[0].apply(lambda x: i if x == uni[i] else x)
        return y

def data_scale(x):
    x = x.apply(pd.to_numeric, errors='ignore')
    numeric_col = x.select_dtypes(include='number')
    numeric_feature_name = numeric_col.columns
    x[numeric_feature_name] = scale.fit_transform(x[numeric_feature_name])
    return x

def dimention_reduction(x):
    x = pca.fit_transform(x)
    return x

classification_models = {
                           "LogisticRegression" : {
                                                        'model' : LogisticRegression(),
                                                        'para' : {
                                                                      'penalty': ['l2'],
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
    model = ""
    first = True
    for model_name,model in classification_models.items():
        grid = RandomizedSearchCV(model['model'],model['para'],cv=3,n_jobs=-1,scoring='accuracy')
        grid.fit(x,y.values.ravel())
        print(f'{model_name} DONEEEEEEEEE')
        print(f"score : {grid.best_score_}")
        if first:
            score = grid.best_score_
            m_name = model_name
            model = grid.best_estimator_
            first = False
        else:
            if grid.best_score_ > score:
                score = grid.best_score_
                m_name = model_name
                model = grid.best_estimator_
        
    print(f"best model : {m_name}")
    print(f"best score : {score}")
    return m_name , score , model
