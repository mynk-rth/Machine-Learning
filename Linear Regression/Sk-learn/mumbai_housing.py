import pandas as pd 
import numpy as np 


# ---------------------------------------------- Train-Test Split -------------------------------------------
def test_train_splits(data_x,data_y,train_size=0.8):
    L = len(data_x)
    train_size = int(L*train_size)
    return [data_x.loc[0:train_size],data_x.loc[train_size:L],data_y.loc[0:train_size],data_y.loc[train_size:L]]


# ---------------------------------------------- Find Categorical Columns -------------------------------------------
def find_cat_cols(data):
    cat_cols = data.select_dtypes(include='str').columns
    return cat_cols.tolist()


# ---------------------------------------------- Encoding Mapping -------------------------------------------
mappings = {}

# ---------------------------------------------- Encoder -------------------------------------------
def encoder(data, col_list):
    global mappings
    # Mapping sirf first time banegi
    if not mappings:
        for col in col_list:
            T = sorted(data[col].unique())
            mapping = {}
            for i, V in enumerate(T):
                mapping[V] = i
            mappings[col] = mapping

    # Old mapping apply hogi
    for col in col_list:
        data[col] = data[col].map(mappings[col])


# ---------------------------------------------- Super Encoder -------------------------------------------
def superencoder(data,target):
    target_column = target
    num_cols = data.select_dtypes(include=['int', 'float']).columns.drop(target_column, errors='ignore')
    cat_cols = data.select_dtypes(include='str').columns.drop(target_column, errors='ignore')
    
    data = data.dropna(subset=num_cols.tolist() + cat_cols.tolist() + [target_column])

    encoder(data,cat_cols)

    Data_X = data[num_cols.tolist() + cat_cols.tolist()]
    Data_y = data[target_column]

    return Data_X,Data_y


#---------------------------------------------- Data Fetching -------------------------------------------
data = pd.read_csv(f"M:/DISK M/Python~SS/Machine Learning/Linear Regression/Sk-learn/mumbai_housing.csv")
# print(data)
# print(data.columns)


#---------------------------------------------- Remove Unnecssary column -------------------------------------------
data = data.drop(['title','price_per_sqft','city','latitude','longitude'],axis=1)


# #---------------------------------------------- Target, Numeric & Categorical -------------------------------------------
# print(data.dtypes)

# target_column = "price"

# num_cols = data.select_dtypes(include=['int', 'float']).columns.drop(target_column, errors='ignore')
# # print("Numerical Columns:", num_cols)

# cat_cols = data.select_dtypes(include='str').columns.drop(target_column, errors='ignore')
# # print("Categorical Columns:", cat_cols)


# #---------------------------------------------- Data Cleaning -------------------------------------------

# data = data.dropna(subset=num_cols.tolist() + cat_cols.tolist() + [target_column])


# #---------------------------------------------- Data Encoding -------------------------------------------
# encoder(data,cat_cols)
# # print(data)


# #---------------------------------------------- Separate Data_X & Data_Y -------------------------------------------
# Data_X = data[num_cols.tolist() + cat_cols.tolist()]
# # print(Data_X)

# Data_y = data[target_column]
# # print(Data_y)


#---------------------------------------------- Single Function --> Def superencoder -------------------------------------------
Data_X, Data_Y = superencoder(data,"price")
# print(Data_X)
# print(Data_Y)


#---------------------------------------------- train-test splits -------------------------------------------
x_train,x_test,y_train,y_test = test_train_splits(Data_X,Data_Y,0.8)  

# print(x_train)
# print(y_train)

# print(x_test)
# print(y_test)



#                                                   ---------------- Calculations --------------
X = x_train.copy()                                                                                             #X = np.column_stack([np.ones(len(x_train)), x_train])
X.insert(0,"intercept",1)
# print(X)

X_transpose = X.T                                                                           # same in numpy & pandas 
# print(X_transpose)

XTX_inv = np.linalg.inv(X_transpose @ X)                                                            # no metric_inverse function in pandas 
# print(XTX_inv)

A_cap = (XTX_inv.dot(X_transpose)).dot(y_train)                                                       #A_cap = (XTX_inv @ X_transpose) @ y_train
# print("A_cap : ",A_cap) 


#                              ---------------- Intercept(a0) & Slope(a1, a2 ,a3,..) --------------
a0 = A_cap[0]
# print("\nIntercept :", a0)

# for i in range(1, len(A_cap)):
#     print(f"Slope a{i} ({Data_X.columns[i-1]}) :", A_cap[i])



#                                                   ---------------- New House ----------------
new_house = pd.DataFrame({
    "area": [1200],
    "bedroom_num": [2],
    "bathroom_num": [2],
    "balcony_num": [1],
    "age": [5],
    "total_floors": [10],
    "locality": ["Kalyan"],
    "property_type": ["Apartment"],
    "furnished": ["Furnished"]
})

# print("\nNew House Before Encoding :")
# print(new_house)

# ---------------------------------------------- Encoding New House -------------------------------------------
# Mapping pass nahi karni hai
# Automatically old training mapping use hogi

categorical_columns=find_cat_cols(new_house)
encoder(new_house,categorical_columns)

# print("\nNew House After Encoding :")
# print(new_house)


# ---------------------------------------------- Fix order of column -------------------------------------------
new_house = new_house[Data_X.columns]

# print("\nNew House With Same Feature Order :")
# print(new_house)


# ---------------------------------------------- Prediction Matrix -------------------------------------------
X_new = np.column_stack([
    np.ones(len(new_house)),
    new_house
])

# print("\nPrediction Matrix :")
# print(X_new[0])


# ---------------------------------------------- Prediction Formula -------------------------------------------
# prediction = A_cap[0]

# for i in range(1, len(A_cap)):
#     prediction = (prediction + A_cap[i] * X_new[0][i])                                            # or A_cap[i] * new_house.iloc[0][i-1]

# 2nd method 
prediction = X_new @ A_cap
print("\nPredicted Price :", prediction)


# # ---------------------------------------------- MSE -------------------------------------------
# intercept = A_cap[0]
# y_pred = pd.Series(intercept,index=x_test.index)
# for i, column in enumerate(x_test.columns):
#     y_pred = (y_pred + A_cap[i + 1] * x_test[column])

# 2nd method
X_test = x_test.copy()
X_test.insert(0, "intercept", 1)

y_pred = X_test @ A_cap
diff = y_test - y_pred
sq = diff ** 2
mse = sq.mean()                                                                                         #sq.sum()/n        where n-> len(x_test)
print("MSE :", mse)                                         

