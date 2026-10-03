import pandas as pd
import numpy as np


class Multi_LR:
  def __init__(self):
        self.mappings = {}

  # ---------------------------------------------- Train-Test Split -------------------------------------------
  def test_train_splits(self,data_x,data_y,train_size=0.8):
    L = len(data_x)
    train_size = int(L*train_size)
    return [data_x.loc[0:train_size],data_x.loc[train_size:L],data_y.loc[0:train_size],data_y.loc[train_size:L]]


  # ---------------------------------------------- Encoder -------------------------------------------

  def onehot_encoder(self,data, col_list=None):
    if col_list is None:
      col_list = self.cat_cols.tolist()
    data = pd.get_dummies(dtype=int,data=data,columns=col_list)
    return data

  # ---------------------------------------------- Super Encoder -------------------------------------------
  def datapreprocessing(self,data,target):
    target_column = target
    num_cols = data.select_dtypes(include=['int', 'float']).columns.drop(target_column, errors='ignore')
    self.cat_cols = data.select_dtypes(include='str').columns.drop(target_column, errors='ignore')
    
    data = data.dropna(subset=num_cols.tolist() + self.cat_cols.tolist() + [target_column])


    Data_X = data[num_cols.tolist() + self.cat_cols.tolist()]
    Data_y = data[target_column]

    Data_X = self.onehot_encoder(Data_X)

    return Data_X,Data_y


  def fit(self,x_train,y_train):
    self.x_train = x_train
    X = x_train.copy()
    X.insert(0,"intercept",1)
    X_transpose = X.T  
    XTX_inv = np.linalg.pinv(X_transpose @ X)
    self.A_cap = (XTX_inv.dot(X_transpose)).dot(y_train)

    print("\n\t\tModel Training Completed\n")

  def predict(self,new_input):
    new_input = new_input.copy()
    
    # Training ke same columns/order mein lao
    new_input = new_input.reindex(
        columns=self.x_train.columns,
        fill_value=0
    )

    new_input.insert(0,"intercept",1)
    prediction = new_input.dot(self.A_cap)
    return prediction.to_numpy()

  def MSE(self,y_test,y_pred):
    diff = y_test - y_pred
    sq = diff ** 2
    mse = sq.mean()                                                                                         #sq.sum()/n        where n-> len(x_test)
    return mse 
  

