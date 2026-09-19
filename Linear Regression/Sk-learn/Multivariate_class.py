import pandas as pd
import numpy as np


class Multivariate_LR:
  def __init__(self):
        self.mappings = {}

  # ---------------------------------------------- Train-Test Split -------------------------------------------
  def test_train_splits(self,data_x,data_y,train_size=0.8):
    L = len(data_x)
    train_size = int(L*train_size)
    return [data_x.loc[0:train_size],data_x.loc[train_size:L],data_y.loc[0:train_size],data_y.loc[train_size:L]]


  # ---------------------------------------------- Find Categorical Columns -------------------------------------------
  def find_cat_cols(self,data):
    cat_cols = data.select_dtypes(include='str').columns
    return cat_cols.tolist()


  # ---------------------------------------------- Encoder -------------------------------------------
  def encoder(self,data, col_list):
    global mappings
    # Mapping sirf first time banegi
    if not self.mappings:
        for col in col_list:
            T = sorted(data[col].unique())
            mapping = {}
            for i, V in enumerate(T):
                mapping[V] = i
            self.mappings[col] = mapping

    # Old mapping apply hogi
    for col in col_list:
        data[col] = data[col].map(self.mappings[col])


  # ---------------------------------------------- Super Encoder -------------------------------------------
  def superencoder(self,data,target):
    target_column = target
    num_cols = data.select_dtypes(include=['int', 'float']).columns.drop(target_column, errors='ignore')
    cat_cols = data.select_dtypes(include='str').columns.drop(target_column, errors='ignore')
    
    data = data.dropna(subset=num_cols.tolist() + cat_cols.tolist() + [target_column])

    self.encoder(data,cat_cols)

    self.Data_X = data[num_cols.tolist() + cat_cols.tolist()]
    Data_y = data[target_column]

    return self.Data_X,Data_y


  def fit(self,x_train,y_train):
    X = x_train.copy()
    X.insert(0,"intercept",1)
    X_transpose = X.T  
    XTX_inv = np.linalg.inv(X_transpose @ X)
    self.A_cap = (XTX_inv.dot(X_transpose)).dot(y_train)

    print("\n\t\tModel Training Completed\n")

  def predict(self,new_input):

    categorical_columns=self.find_cat_cols(new_input)            # new_house ke categorical columns find kiye 
    self.encoder(new_input,categorical_columns)                  # encode kar diya categorical columns ko
    new_input = new_input[self.Data_X.columns]                         # columns ko sahi order me fix kiya just like A_cap
    X_new = np.column_stack([
        np.ones(len(new_input)),                                  # matric create kiya 
        new_input
        ])

    prediction = self.A_cap[0]
    for i in range(1, len(self.A_cap)):
        prediction = (prediction + self.A_cap[i] * X_new[0][i])
    return prediction 

  def MSE(self,x_test,y_test):
    intercept = self.A_cap[0]
    y_pred = pd.Series(intercept,index=x_test.index)

    for i, column in enumerate(x_test.columns):
        y_pred = (y_pred + self.A_cap[i + 1] * x_test[column])

    diff = y_test - y_pred
    sq = diff ** 2
    mse = sq.mean()                                                                                         #sq.sum()/n        where n-> len(x_test)
    return mse 

