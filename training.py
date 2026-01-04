from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB , GaussianNB
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


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

regression_models = {
                           "LinearRegression" : {
                                                        'model' : LinearRegression(),
                                                        'para' : {
                                                                      'fit_intercept': [True,False]
                                                        }
                           },
                           "Lasso" : {
                                                        'model' : Lasso(),
                                                        'para' : {
                                                                  'alpha' : [0.001,0.01,0.1,1,10],
                                                                  'max_iter':  [200,500,1000]  
                                                        }
                           },
                           "Ridge" : {
                                                        'model' : Ridge(),
                                                        'para' : {
                                                                  'alpha' : [0.001,0.01,0.1,1,10],
                                                                  'max_iter':  [200,500,1000]  
                                                        }
                           },
                            "RandomForestRegressor" : {
                                                        'model' : RandomForestRegressor(),
                                                        'para' : {
                                                                'n_estimators': [200,300],
                                                                'max_depth': [10, 20, None],
                                                                'min_samples_leaf': [1, 5, 10],
                                                                'max_features': ['sqrt', 'log2']
                                                                }
                             },
                             "KNN" : {
                                        'model' : KNeighborsRegressor(),
                                        'para' : {
                                                    'n_neighbors': range(1, 5)
                                                 }
                            }
}

def c_train(x,y):           #training classification model 
    model = ""
    first = True
    for model_name,model in classification_models.items():
        grid = RandomizedSearchCV(model['model'],model['para'],cv=3,n_jobs=-1,scoring='accuracy')
        grid.fit(x,y.values.ravel())
        print(f'{model_name} DONEEEEEEEEE')
        print(f"score : {grid.best_score_}")
        if first:
            score = grid.best_score_            #initilizing score
            m_name = model_name                 #initilizing model name
            model = grid.best_estimator_        #initilizing best model
            first = False
        else:
            if grid.best_score_ > score:            #compareing model scores
                score = grid.best_score_            #updating score
                m_name = model_name                 #updating model name
                model = grid.best_estimator_        #updating best model
        
    print(f"best model : {m_name}")
    print(f"best score : {score}")
    return m_name , score , model

def r_train(x,y):               #training classification model 
    model = ""
    first = True
    for model_name,model in regression_models.items():
        grid = GridSearchCV(model['model'],model['para'],cv=3,n_jobs=-1,scoring='r2')
        grid.fit(x,y.values.ravel())
        print(f'{model_name} DONEEEEEEEEE')
        print(f"score : {grid.best_score_}")
        if first:
            score = grid.best_score_            #initilizing score
            m_name = model_name                 #initilizing model name
            model = grid.best_estimator_        #initilizing best model
            first = False
        else:
            if grid.best_score_ > score:        #compareing model scores
                score = grid.best_score_        #updating score
                m_name = model_name             #updating model name
                model = grid.best_estimator_    #updating best model
        
    print(f"best model : {m_name}")
    print(f"best score : {score}")
    return m_name , score , model
