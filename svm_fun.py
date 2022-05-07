from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import GridSearchCV
 

from systemMatrix import *


#data-prep
train_data = data[0:6] 
features = []
y = []
num_patients = 4
patients_to_predict = []
pred_data = data[6:(6+num_patients)]
#create feature-matrix - every tissue-curve in every region at every time is considered feature
for i in range(6):
    for j in range(train_data.shape[1]):
        print(train_data[i][j][1:6])
        features.append(train_data[i][j][1:6])
        if(i<=2): #create resultant vector
            y.append(0) #label all healthy entries
        else:
            y.append(1) #label all sick entries
for i in range(4):
    for j in range(pred_data.shape[1]):
        patients_to_predict.append(pred_data[i][j][1:6])

'''features = np.array(features) #a bit stupid, but needs list of lists-format
features_norm = preprocessing.normalize(features, axis=0)
list_of_lists = features.tolist() #for some reason needs to be list of lists
list_of_lists_norm = features_norm.tolist()
patients_to_predict_l = np.array(patients_to_predict).tolist()
patients_to_predict_l = preprocessing.normalize(patients_to_predict_l,axis=0)

#generate machine 
clf = SVC(kernel='linear')
clf.fit(list_of_lists_norm,y)
y_pred = clf.predict(list_of_lists_norm)
#print(accuracy_score(y, y_pred)) #accuracy score 
#cross-validation for best estimate of generalization error
scores = cross_val_score(clf, list_of_lists_norm, y, cv=10) #get accuracy (fine for our dataset, as it is fully balanced)
print(scores)

new_preds = clf.predict(patients_to_predict_l) #get oredictions on the "new" dataset
holder = np.array(new_preds)

verdict = []
for i in range(num_patients):
    verdict.append(stats.mode(holder[i*158:i*158+158])) #get mode of SVM-prediction
    print(verdict[i])
    if(verdict[i][0]==0):
        print("Patient {} is healthy".format(i+1))
    elif(verdict[i][0]==1):
        print("Patient {} is sick".format(i+1))
    #del verdict
    
    
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# fitting the model for grid search
grid.fit(list_of_lists_norm,y)
g1_o_pred = grid.predict(list_of_lists_norm)
g1_n_pred = grid.predict(patients_to_predict_l)'''

#alternative approach - see all time-data points as a entry only "together". Thus, loosing time-dimension, but comparing all data-points
pooled_measurements = []
pooled_predictor = []
for i in range(train_data.shape[0]):
    norm_vec = preprocessing.normalize(train_data[i]).reshape(-1)
    pooled_measurements.append(norm_vec.tolist())
    

for i in range(data.shape[0]-train_data.shape[0]):
    norm_vec = preprocessing.normalize(pred_data[i]).reshape(-1)
    pooled_predictor.append(norm_vec)

pooled_y = np.array([0,0,0,1,1,1])
clf2 = SVC(kernel='linear',C=1,probability=(True))
clf2.fit(pooled_measurements,pooled_y)
org_preds = clf2.predict(pooled_measurements)
new_preds = clf2.predict(pooled_predictor) #get oredictions on the "new" dataset

print(new_preds)
#estimate generalization ability
scores = cross_val_score(clf2,pooled_measurements,org_preds,cv=3)
print(scores)


#using parameter-fitting - should not be used, please
# defining parameter range
param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'linear']} 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=2)
# fitting the model for grid search
grid.fit(pooled_measurements,pooled_y)
g2_o_pred = grid.predict(pooled_measurements)
g2_n_pred = grid.predict(pooled_predictor)

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
