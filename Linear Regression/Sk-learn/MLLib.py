import pandas as pd
def test_train_splits(data_x,data_y):
    L=len(data_x)
    train_size=int(L*0.8)
    #test_size=int(L*0.2)
    return [data_x.loc[0:train_size],data_y.loc[0:train_size],data_x.loc[train_size:L],data_y[train_size:L]]


def encoder(data,columns):
   data=data.copy()
   data=data[columns] 
   data = pd.get_dummies(data, columns=columns,dtype=int)
   print(data)
   encoder_columns=data.columns
   print(encoder_columns)
   return(data,encoder_columns)
   
def encode_column(data,col_list):
  for col in col_list:
    T=data[col].unique()
    T.sort()
    mapping={}
    for i,V in enumerate(T):
       mapping[V]=i
    data[col]=data[col].map(mapping)
  return data
