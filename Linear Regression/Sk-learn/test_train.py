import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split


def test_train_splits(data_x,data_y,train_size=0.8):
    L = len(data_x)
    train_size = int(L*train_size)
    return [data_x.loc[0:train_size],data_x.loc[train_size:L],data_y.loc[0:train_size],data_y.loc[train_size:L]]

def encoder(data,col_list):
    for col in col_list:
        T = sorted(data[col].unique())
        mapping = {}
        for i,V in enumerate(T):
            mapping[V]=i
        data[col]=data[col].map(mapping)


data = pd.read_csv(f"M:/DISK M/Python~SS/Machine Learning/Linear Regression/Sk-learn/mumbai_housing.csv")
# print(data)


# print(data.columns)


data_y = data['price']
# print(data_y)


data_x = data.drop(['price'],axis=1)
# print(data_x.columns)
data_x.drop(['title'],axis=1,inplace=True)
# print(data_x)


x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=10)         # sklearn train_test_split function
# print(y_train)


x_train,x_test,y_train,y_test = test_train_splits(data_x,data_y)                        # def test_train_splits function 
# print(y_test)


columns = ['locality', 'city','property_type','furnished']
encoder(data,columns)
# print(data['furnished'])
# print(data['city'])

#-------------------------------------------------------  Do not use this function  ----------------------------------
data_x = pd.get_dummies(dtype=int,data=data_x,columns=columns,drop_first=True)        # error --> Do not use this function 
print(data_x)
print(data_x.columns)

