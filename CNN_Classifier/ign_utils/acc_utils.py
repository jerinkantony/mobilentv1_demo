

#
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
import numpy as np

def acc_matrics(y_true, y_pred,labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print('Confusion matrix:\n',cm)
    print('Report :\n',classification_report(y_true, y_pred))
    #import pdb;pdb.set_trace()
    goodcount=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            goodcount+=1
    
    print('Val Acc: ', goodcount/len(y_true) * 100, '%\n\n')
    #print('Accuracy Score :\n',accuracy_score(y_true, y_pred))

