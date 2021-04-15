import ast, shutil
import numpy as np
from nproc import npc
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy import interp
from sklearn.metrics import roc_curve, roc_auc_score
import pylab as pl
from sklearn.metrics import confusion_matrix


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
      
    print('- Accuracy Avg:')
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    
    
    for model in models_cam:
        
        test3 = 0
        print()
        print()
        print(model)
        
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
            
            print()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        
            test = npc()

            fit = test.npc(X_train, y_train, 'nb', alpha=0.01, delta=0.08, n_cores=os.cpu_count())
            
            fitted_score = test.predict(fit, X_train)            
            acc_train = accuracy_score(fitted_score[0], y_train)
            
            
            pred_score = test.predict(fit, X_test)
            acc_test = accuracy_score(pred_score[0], y_test)
            
            
            fpr, tpr, thresholds = roc_curve(y_test, pred_score[1])
            
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = roc_auc_score(pred_score[0], y_test)

            pl.plot(fpr, tpr, lw=1, label='ROC fold-%d (AUC = %0.2f)' % (i, roc_auc))
            
            cm = confusion_matrix(y_true=y_test, y_pred=pred_score[0], labels=[1, 0])
            
            print(cm)
            print("Training set:", round(acc_train, 2))
            print("Test set:", round(acc_test, 4))
            test3 += acc_test
            
            
            i += 1

        pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('FPR')
        pl.ylabel('TPR')
        pl.title(f'Curva ROC ({model})')
        pl.legend(loc="lower right")
        pl.show()
        print(round(test3 * 100/5, 4))
    #print(models_cam)
    
            
        
    
        
        
  
    
        
    
    