import pandas as pd
import numpy as np
import pickle
data=pd.read_excel("D:/iris_flask/iris .xls")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['Classification']=le.fit_transform(data['Classification'])

y=data['Classification']

x=data.drop(['Classification'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=25)

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
lr =LogisticRegression(max_iter = 250)
model = lr.fit(x_train,y_train)

pickle_file = 'lr_model.pickle'
with open(pickle_file,'wb') as file:
    pickle.dump(model,file)