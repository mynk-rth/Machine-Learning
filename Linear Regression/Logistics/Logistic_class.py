import pandas as pd
import numpy as np


class Logistic_Regression:
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
  def encoder(self,data, col_list=None):

    if col_list is None:
      col_list = self.cat_cols.tolist()

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
  def datapreprocessing(self,data,target):
    target_column = target
    num_cols = data.select_dtypes(include=['int', 'float']).columns.drop(target_column, errors='ignore')
    self.cat_cols = data.select_dtypes(include='str').columns.drop(target_column, errors='ignore')
    
    data = data.dropna(subset=num_cols.tolist() + self.cat_cols.tolist() + [target_column])

    self.encoder(data)

    self.Data_X = data[num_cols.tolist() + self.cat_cols.tolist()]
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
    new_input = new_input.copy()[self.Data_X.columns]
    new_input.insert(0,"intercept",1)
    equation = new_input.dot(self.A_cap)  

    self.estimated_probability = 1 / (1 + np.exp(-equation))                             # Sigmoid

    prediction = (self.estimated_probability >= 0.5).astype(int)                                 # Probability ko class mein convert karna              

    return prediction.to_numpy()



  def MSE(self,y_test,y_pred):
    diff = y_test - y_pred
    sq = diff ** 2
    mse = sq.mean()                                                                                         #sq.sum()/n        where n-> len(x_test)
    return mse 
  






































 # # ---------------------------------------------
    # # Convert input into DataFrame  --> encoder
    # # ---------------------------------------------
    # if isinstance(mydata, dict):
    #     data = pd.DataFrame([mydata])
    # else:
    #     data = mydata.copy()











# def predict(self, new_input):

#     # Agar direct integer/float diya hai
#     if isinstance(new_input, (int, float)):
#         new_input = pd.DataFrame({
#             self.Data_X.columns[0]: [new_input]
#         })

#     # Agar DataFrame diya hai
#     else:

# def accuracy(self, y_test, y_pred):

#     y_test = np.asarray(y_test)
#     y_pred = np.asarray(y_pred)

#     return np.mean(y_test == y_pred)