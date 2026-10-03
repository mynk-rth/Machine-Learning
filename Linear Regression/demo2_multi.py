import pandas as pd
import numpy as np
from Multi_class import Multi_LR


#---------------------------------------------- Data Fetching -------------------------------------------
data = pd.read_csv(f"M:/DISK M/Python~SS/Machine Learning/Linear Regression/Datasets/mumbai_housing.csv")
# print(data)
# print(data.columns)

#---------------------------------------------- Remove Unnecssary columns -------------------------------------------
data = data.drop(['title','price_per_sqft','latitude','longitude'],axis=1)


#---------------------------------------------- Model -------------------------------------------
Model = Multi_LR()

Data_X, Data_Y = Model.datapreprocessing(data,"price")                            # Data , target_column(data_Y column)

x_train,x_test,y_train,y_test = Model.test_train_splits(Data_X,Data_Y,0.8)  

Model.fit(x_train,y_train)

new_house = pd.DataFrame({
    "area": [200],
    "bedroom_num": [3],
    "bathroom_num": [3],
    "balcony_num": [2],
    "age": [2],
    "total_floors": [1],
    "city": ["Mumbai"],
    "locality": ["Andheri West"],
    "property_type": ["Apartment"],
    "furnished": ["Furnished"]
})
new_house=Model.onehot_encoder(new_house)                                      # encode kar diya categorical columns ko

print("Prediction of New House: ",Model.predict(new_house)[0])


# ---------------------------------------------- MSE -------------------------------------------

y_pred = Model.predict(x_test)
print("Prediction of x_test : \n",y_pred)
print(type(y_pred))
print("MSE : ",Model.MSE(y_test,y_pred))



# SHOW ACTUAL VS PREDICTED

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred.round(2)
})

print("\nActual vs Predicted:")

print(result.head(10))














