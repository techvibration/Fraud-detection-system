import numpy as np
import pandas as pd
import xgboost as xgb
#print("XGBoost version:", xgb.__version__)
from imblearn.combine import SMOTETomek
import optuna   
import joblib
from xgboost import XGBClassifier
from xgboost import callback
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score,cross_val_predict,RepeatedStratifiedKFold 
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

dataset_path = "D:\\Fraud-detection-system\\data\\creditcard.csv" 
fds_dataset = pd.read_csv(dataset_path)

#making the different datasets
y = fds_dataset['Class']
X = fds_dataset.drop('Class',axis = 1)
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.2,stratify = y,random_state = 42) 

test_X.to_csv('X_val1.csv', index=False)
test_y.reset_index(drop=True).to_frame(name='Class').to_csv('y_val1.csv', index=False)

#using scale_pos_weight to increase recall
scale_pos_weight = (train_y == 0).sum() / (train_y == 1).sum() 
adjusted_scale_pos_weight = scale_pos_weight * 2.5
cv = RepeatedStratifiedKFold(n_splits = 5,n_repeats = 10,random_state = 42)
fd_model = XGBClassifier(use_label_encoder = False,eval_metric = 'logloss',scale_pos_weight = adjusted_scale_pos_weight)
fd_model.fit(train_X,train_y,eval_set =[(test_X,test_y)],verbose = True)

CV_scores = cross_val_score(fd_model,train_X,train_y,cv = cv,scoring = 'roc_auc')
print("Cross Validation score",CV_scores)
print("Mean Cross Validation score",np.mean(CV_scores))

predictions = fd_model.predict(test_X)
probablity = fd_model.predict_proba(test_X)[:,1]
#setting threshold to increase recall of the failure class
threshold = 0.2
custom_predictions = (probablity >= threshold).astype(int)
#printing the classification report for precision,recall,f1score
print("Classification Report:")
print(classification_report(test_y,custom_predictions))
#printing the confusion matrix
print("Confusion Matrix:")
print (confusion_matrix(test_y,custom_predictions))
#Calculate the roc-aur score
roc_auc = roc_auc_score(test_y,custom_predictions)
print("ROC-AUC score:",roc_auc)
joblib.dump(fd_model,'model_main.pkl')
print("model saved as model_main.pkl") 






