# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:29:26 2019

@author: ASUS
"""

import pickle
with open("2019-05-16_19-03-29-1.pickle", "rb") as f:
    data_list1 = pickle.load(f)
import pickle
with open("2019-05-16_19-04-41-2.pickle", "rb") as f:
    data_list2 = pickle.load(f)
import pickle
with open("2019-05-16_19-06-39-3.pickle", "rb") as f:
    data_list3 = pickle.load(f)
import pickle
with open("2019-05-16_19-16-58-6.pickle", "rb") as f:
    data_list4 = pickle.load(f)
import pickle
with open("2019-05-16_19-25-51-9.pickle", "rb") as f:
    data_list5 = pickle.load(f)

data_list = data_list1 + data_list2 + data_list3
# save each information seperately
Ballposition=[]
Bricks=[]
Frame=[]
PlatformPosition=[]
Status=[]

for i in range(0, len(data_list)):
    Ballposition.append(data_list[i].ball)
    Bricks.append(data_list[i].bricks)
    Frame.append(data_list[i].frame)
    PlatformPosition.append(data_list[i].platform)
    Status.append(data_list[i].status)
    
#%% calculate instruction of each frame using platformposition
import numpy as np
PlatX = np.array(PlatformPosition)[:,0][:, np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

# select some features to make x
Ballarray = np.array(Ballposition[:-1]) 
x = np.hstack((Ballarray, PlatX[0:-1,0][:,np.newaxis]))
# select intructions as y
y = instruct.ravel()


#%% train your model here
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.01, random_state = 41)
model = SVR(gamma=0.08, C=0.01, epsilon=1.0)
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
# check the acc to see how well you've trained the model
MSE = mean_squared_error(y_predict, y_test)
RMSE = np.sqrt(MSE)

import pickle
filename = "model.sav"
pickle.dump(model, open(filename, 'wb'))

# load model
l_model = pickle.load(open(filename,'rb'))
yp_l = l_model.predict(x_test)
MSE = mean_squared_error(yp_l, y_test)
RMSE = np.sqrt(MSE)
print("RMSE=%f" % (RMSE))

