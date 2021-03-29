import ast, shutil
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nproc import npc
import os
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.metrics import roc_curve, auc
import pylab as pl


def get_corrs_data(location):
    with open(f'{location}', 'r') as f:
        corr = f.read().split('\n')
        corr = [*map(lambda x: x.split(', ')[0], corr)]
        corr = corr[0:-1]
        corr = np.array([*map(lambda x: ast.literal_eval(x), corr)])
    return corr
    

def experiment(models_cam):
    
    shutil.rmtree('./model/', ignore_errors=True)

    if not os.path.exists('./model/'):
        os.makedirs('./model/')
      
    #print('- Accuracy Avg:')
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    
    for model in models_cam:
        
        diff = get_corrs_data(f'./correlation/{model}_not_taken.txt')
        same = get_corrs_data(f'./correlation/{model}_taken.txt')
        
        np.random.shuffle(diff)
        #diff = diff[np.random.choice(len(diff), size=same.shape[0], replace=False)]
        np.random.shuffle(same)
        
        diff_labeled = np.vstack((diff, np.zeros(diff.shape[0])))
        same_labeled = np.vstack((same, np.ones(same.shape[0])))

        dataset = np.hstack((same_labeled, diff_labeled)).T
        np.random.shuffle(dataset)
        
        X = dataset[:,0]
        y = dataset[:,1]
 
        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        i = 0
        for train_index, test_index in skf.split(X, y):
            
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        
            test = npc()

            fit = test.npc(X_train, y_train, 'nb', n_cores=os.cpu_count())
            pred_score = test.predict(fit, X_test)
            
            fpr, tpr, thresholds = roc_curve(y_test, pred_score[1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            
            i += 1
        
        print(model)
        print(roc_auc)
        pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title(f'ROC Curve ({model})')
        pl.legend(loc="lower right")
        pl.show()
            
        
    
        
        
  
    
        
    
    