# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:34:07 2019

@author: Sajal
"""

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
np.random.seed(7)

Data=sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
Label=sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

Data=np.reshape(Data,(Data.shape[0]*Data.shape[1],Data.shape[2]))

Label=np.reshape(Label,(Label.shape[0]*Label.shape[1]))

Labels,counts=np.unique(Label,return_counts=True)

Data=Data[Label>0,:]
Label=Label[Label>0]

Labels,counts=np.unique(Label,return_counts=True)


Labels,counts=np.unique(Label,return_counts=True)

#Standardizing the values 
"""
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Data) 
Data= scaler.transform(Data)

"""

#Train-Test Split

from sklearn.model_selection import train_test_split

testRatio=0.20

X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=testRatio, random_state=345,
                                                        stratify=Label)



#Applying Scalar to train and test Dataset

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train) 
X_train= scaler.transform(X_train)
X_test=scaler.transform(X_test)







#Applying PCA 


import matplotlib.pyplot as plt
pca = PCA(n_components=30)
pca.fit_transform(X_train)
newspace=pca.components_
newspace=newspace.transpose()
X_train=np.matmul(X_train,newspace)
print(pca.n_components_)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Inidian_pines Dataset Explained Variance')
plt.show()
X_test=np.matmul(X_test,newspace)


#Appplying SVM 

from sklearn.svm import SVC

clf = SVC(C=10,gamma=0.01)

clf.fit(X_train, y_train)

#model tuning

import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


start=time.time()
C_range = np.logspace(-2, 10, 13)
gamma_range=np.logspace(-9, 3, 13)
param_grid = dict(C=C_range,gamma=gamma_range)
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
grid = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy', return_train_score=False)
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
end=time.time()
print(end-start)



import pickle
pickle_out = open("Firstgrid16.pickle","wb")
pickle.dump(grid, pickle_out)
pickle_out.close()

pickle_in = open("Firstgrid16.pickle","rb")
example_dict = pickle.load(pickle_in)
example_dict.best_estimator_

#Predicting accuracy
y_pred=clf.predict(X_test) 
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


from sklearn.metrics import cohen_kappa_score
print(cohen_kappa_score(y_test,y_pred))

from sklearn.metrics import cohen_kappa_score
print(cohen_kappa_score(y_test,y_pred))



