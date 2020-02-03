# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:29:26 2019

@author: ASUS
"""

import pickle
with open("2019-04-12_09-12-23-1.pickle", "rb") as f:
    data_list1 = pickle.load(f)
import pickle
with open("2019-04-12_09-12-46-2.pickle", "rb") as f:
    data_list2 = pickle.load(f)
import pickle
with open("2019-04-12_09-13-17-3.pickle", "rb") as f:
    data_list3 = pickle.load(f)

data_list = data_list1+data_list2+data_list3

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.01, random_state = 3)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)
yt = model.predict(x_test)
# check the acc to see how well you've trained the model
acc = accuracy_score(yt, y_test)

import pickle
filename = "model.sav"
pickle.dump(model, open(filename, 'wb'))

# load model
l_model = pickle.load(open(filename,'rb'))
yp_l = l_model.predict(x_test)
print("acc load: %f " % accuracy_score(yp_l, y_test))

