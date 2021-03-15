import ast
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from .correlation import calc_correlation
from nproc import npc
import os
import pickle
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def get_corrs_data(location):
    with open(f'{location}', 'r') as f:
        corr = f.read().split(', ')
        corr = corr[0:-1]
        corr = np.array([*map(lambda x: ast.literal_eval(x), corr)])
    return corr
    

def extract_threshold(models_cam):
    
    path_corr = './model/'

    if not os.path.exists(path_corr):
        os.makedirs(path_corr)
        
    for model in models_cam:
        print(model)
        same = get_corrs_data(f'./correlation/{model}_taken.txt')
        diff = get_corrs_data(f'./correlation/{model}_notTaken.txt')
        
        
        same_labeled = np.vstack((same, np.ones(same.shape[0])))
        diff_labeled = np.vstack((diff, np.zeros(diff.shape[0])))
        dataset = np.hstack((same_labeled, diff_labeled)).T
        np.random.shuffle(dataset)
        print(dataset[0:10])
        
        X = dataset[:,0]
        y = dataset[:,1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        
        test = npc()
        
        '''
        method  	logistic: Logistic Regression.
    		svm: Support Vector Machine.
    		nb: Gaussian Naive Bayes.
    		nb_m: Multinomial Naive Bayes.
    		rf: Random Forest.
    		dt: Decision Tree.
        '''
        fit = test.npc(X_train, y_train, 'rf', n_cores=os.cpu_count())

        fitted_score = test.predict(fit, X_train)
        acc_train = accuracy_score(fitted_score[0], y_train)
        print(" - Accuracy on training set:", acc_train)
    
        pred_score = test.predict(fit, X_test)
        acc_test = accuracy_score(pred_score[0], y_test)
        print(" - Accuracy on test set:", acc_test)
        
        cm = confusion_matrix(y_test, pred_score[0])
        print(f" - Confusion matrix: {cm.tolist()}")
        accuracy_score(pred_score[0], y_test)
        tn, fp, fn, tp = cm.ravel()
        print(" - Type I error rate: {:.5f}".format(fp/(fp+tn)))
        print(" - Type II error rate: {:.5f}\n".format(fn/(fn+tp)))
        
        pickle.dump(fit, open(f'./model/{model}', 'wb'))
        
    
        
        
  
    
        
    
    