# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:02:06 2018

@author: supinya

Functions:    
    plot_features(X_columns, model, save_fig=False, figtitle = 'untitled')  
    classifier_performance(X_train, Y_train, X_test, Y_test, model)
    regressor_performance(X_train, Y_train, X_test, Y_test, model)
    pickle_cv_results(file_name, cv_results)
    unpickle_cv_results(file_name)
    pickle_model(file_name, model, cv_results)
    unpickle_model(file_name)
"""
###############################################################################
#IMPORT PACKAGES
###############################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer as Imputer
import pickle
import joblib

###############################################################################
# DEFINE FUNCTIONS
###############################################################################
# drop non-numeric and other descriptive columns, fill in missing values with either 0 (imputer=False) or the mean (inputer=True)
def generate_X(compounds, impute=False):        ##Note: only impute median/mean from beers with same tasting category? Or take 3 nearest neighbors and impute from there.
    index_ = compounds.beer_id
    col_drop = ['date','beer_id', 'ABV','Barcode', 'beer', 'brewery', 'tasting_category']
    X = compounds
    for col in col_drop:
        if col in X.columns:
            X = X.drop(col,axis =1)
    if 'CO2-PSI.' in X.columns:
        X = X.drop(['CO2-PSI.'],axis = 1)
    if(impute == True):
        #cols_X = X.columns
        imputer = Imputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=index_)
    else:
        X.fillna(value=0,axis=1,inplace =True)
    X.set_index(index_, inplace=True)
    return X

# Visualize how well spread train and test sets are - use LDA with color labels = test/train
def train_test_spread(X, X_train, X_test, y_class, method='LDA'):
    import matplotlib.pyplot as plt
    from itertools import chain
    if method == 'PCA':
        from sklearn.decomposition import PCA
    else:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    ind_train = list(chain(*[[i for i,x in enumerate(X.index) if x == a] for a in X_train.index]))
    ind_test = list(chain(*[[i for i,x in enumerate(X.index) if x == a] for a in X_test.index]))
    
    from sklearn.preprocessing import StandardScaler
    std_scale = StandardScaler()
    X = pd.DataFrame(std_scale.fit_transform(X), columns=X.columns)
        
    if method == 'PCA':
        pca = PCA(n_components=2, whiten=True)
        X_pca = pca.fit_transform(X)
        
        x = X_pca[ind_train+ind_test,0]
        y = X_pca[ind_train+ind_test,1]
    else:
        lda = LDA(n_components=2, solver='eigen', shrinkage='auto', priors=y_class.value_counts().sort_index())
        X_lda = lda.fit_transform(X, y=y_class)
    
        x = X_lda[ind_train+ind_test,0]
        y = X_lda[ind_train+ind_test,1]
    
    xlabel='Component 1'
    ylabel='Component 2'
    title='Spread of Train and Test Sets'
    classes=['train']*len(ind_train)+['test']*len(ind_test)
    unique_classes = ['train','test']
    legends_name='Train/Test Split'
    colors = ['blue','red']
        
    fig, ax = plt.subplots(1, figsize=(7, 6))
    for i, u in enumerate(unique_classes):
        xi = [x[j] for j  in range(len(x)) if classes[j] == u]
        yi = [y[j] for j  in range(len(x)) if classes[j] == u]
        plt.scatter(xi, yi, s=18, label=str(u), color=colors[i])
    ax.set_position([0.1,0.1,0.85,0.78])  #left, bottom, width, height
    plt.title(title, fontsize=14, fontweight='bold', y=1.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, edgecolor='white', loc='upper right', title=legends_name, borderaxespad=0, fontsize=12, columnspacing=0.1, framealpha=0)

    return x,y,y_class[ind_train+ind_test]

# Show low/no alcohol beers
def low_alcohol_spread(X, X_train, X_test, y_class, method='LDA'):
    import matplotlib.pyplot as plt
    from itertools import chain
    if method == 'PCA':
        from sklearn.decomposition import PCA
    else:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    ind_train = list(chain(*[[i for i,x in enumerate(X.index) if x == a] for a in X_train.index]))
    ind_test = list(chain(*[[i for i,x in enumerate(X.index) if x == a] for a in X_test.index]))
    
    from sklearn.preprocessing import StandardScaler
    std_scale = StandardScaler()
    X = pd.DataFrame(std_scale.fit_transform(X), columns=X.columns)
        
    if method == 'PCA':
        pca = PCA(n_components=2, whiten=True)
        X_pca = pca.fit_transform(X)
        
        x = X_pca[ind_train+ind_test,0]
        y = X_pca[ind_train+ind_test,1]
    else:
        lda = LDA(n_components=2, solver='eigen', shrinkage='auto', priors=y_class.value_counts().sort_index())
        X_lda = lda.fit_transform(X, y=y_class)
    
        x = X_lda[ind_train+ind_test,0]
        y = X_lda[ind_train+ind_test,1]
    
    xlabel='Component 1'
    ylabel='Component 2'
    title='Spread of Train and Test Sets'
    classes=['train']*len(ind_train)+['test']*len(ind_test)
    unique_classes = ['train','test']
    legends_name='Train/Test Split'
    colors = ['blue','red']
        
    fig, ax = plt.subplots(1, figsize=(7, 6))
    for i, u in enumerate(unique_classes):
        xi = [x[j] for j  in range(len(x)) if classes[j] == u]
        yi = [y[j] for j  in range(len(x)) if classes[j] == u]
        plt.scatter(xi, yi, s=18, label=str(u), color=colors[i])
    ax.set_position([0.1,0.1,0.85,0.78])  #left, bottom, width, height
    plt.title(title, fontsize=14, fontweight='bold', y=1.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, edgecolor='white', loc='upper right', title=legends_name, borderaxespad=0, fontsize=12, columnspacing=0.1, framealpha=0)
        
        
# For tree-based models, plot features according to importance       
def plot_features(X_columns, model, save_fig=False, figtitle = 'untitled'):       
    labels = X_columns[np.argsort(-model.feature_importances_)]
    index = np.arange(len(labels))

    fig, ax = plt.subplots(1,figsize=(27, 7))    
    plt.bar(np.arange(len(model.feature_importances_)),model.feature_importances_[np.argsort(-model.feature_importances_)], width=0.8)
    ax.set_position([0.1,0.35,0.88,0.55])  #left, bottom, width, height
    plt.xlabel('Compounds', fontsize=14)
    plt.xlim([-0.5,len(model.feature_importances_)])
    plt.ylabel('Feature Importance Scores', fontsize=15)
    plt.xticks(index, labels, fontsize=8, rotation=90)
    plt.title("Feature Importances", fontsize=20)
    plt.show()
    if (save_fig == True):
        fig.savefig(os.path.join('Figures',figtitle + '_FeatureImportance.png'), dpi=600)

#Print performance of random forest model
def classifier_performance(X_train, Y_train, X_test, Y_test, model, toprint = True):
    y_pred = model.predict(X_train)
    acc_train = accuracy_score(Y_train, y_pred)
    bacc_train = balanced_accuracy_score(Y_train, y_pred)
    
    y_pred = model.predict(X_test)
    acc_test = accuracy_score(Y_test, y_pred)
    bacc_test = balanced_accuracy_score(Y_test, y_pred)
    
    if(toprint == True):
        print('Training set performance:')
        model_prediction = model.predict(X_train)
        tn, fp, fn, tp = confusion_matrix(Y_train, model_prediction).ravel()
        print('Confusion matrix: '+str(model.score(X_train,Y_train)))
        print('True negative: '+str(tn) + '\t False positive: '+str(fp))
        print('False negative: '+str(fn) + '\t True positive: '+str(tp))
        print()
        
        print('Test set performance:')
        model_prediction = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(Y_test, model_prediction).ravel()
        print('Confusion matrix: '+str(model.score(X_test,Y_test)))
        print('True negative: '+str(tn) + '\t False positive: '+str(fp))
        print('False negative: '+str(fn) + '\t True positive: '+str(tp))
        print()
    
    return acc_train, bacc_train, acc_test, bacc_test
    #print(classification_report(y_true, y_pred, target_names=target_names))

# Print regressor performance    
def regressor_performance(X_train, Y_train, X_test, Y_test, model, toprint=True): 
    y_pred = model.predict(X_train)
    mse_train = mean_squared_error(Y_train, y_pred)
    r2_train = r2_score(Y_train, y_pred)
    
    y_pred = model.predict(X_test)
    mse_test = mean_squared_error(Y_test, y_pred)
    r2_test = r2_score(Y_test, y_pred)
    
    if(toprint == True):
        print('Training set performance:')
        print('Mean squared error: '+str(mse_train))
        print('Coefficient of determination (R^2): '+str(r2_train))
        print()
        
        print('Test set performance:')
        print('Mean squared error: '+str(mse_test))
        print('Coefficient of determination (R^2): '+str(r2_test))
        print()

        print('\n')
        
    return mse_train, r2_train, mse_test, r2_test

# Pickle trained models's cross validation results    
def pickle_cv_results(file_name,cv_results):
    fileObject = open(os.path.join('CV_results', file_name+'_CV_results.pkl'),'wb')
    pickle.dump(cv_results,fileObject)   
    fileObject.close()
    cv_results.to_csv(os.path.join('CV_results', file_name+'_CV_results.csv'))

# Unpickle trained models's cross validation results   
def unpickle_cv_results(file_name):
    fileObject = open(os.path.join('CV_results', file_name+'_CV_results.pkl'),'rb')
    cv_results = pickle.load(fileObject) 
    fileObject.close()
    return cv_results

# Pickle trained models
def pickle_model(file_name, model, cv_results):
    joblib.dump(model, os.path.join('models', file_name+'.pkl'))
    pickle_cv_results(file_name, cv_results)
    print(f'{file_name}')
    
# Unpickle trained models
def unpickle_model(file_name):
    model = joblib.load(os.path.join('models', file_name+'.pkl'))
    cv_results = unpickle_cv_results(file_name)
    return model, cv_results





