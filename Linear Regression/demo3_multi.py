import pandas as pd
import numpy as np
from Multi_class import Multi_LR


#---------------------------------------------- Data Fetching -------------------------------------------
data = pd.read_csv(f"M:/DISK M/Python~SS/Machine Learning/Linear Regression/Datasets/housing.csv")
# print(data)
# print(data.columns)


#---------------------------------------------- Model -------------------------------------------
Model = Multi_LR()

Data_X, Data_Y = Model.datapreprocessing(data,'median_house_value')                            # Data , target_column(data_Y column)

x_train,x_test,y_train,y_test = Model.test_train_splits(Data_X,Data_Y,0.8)  

Model.fit(x_train,y_train)

mydata = {
    "longitude": -122.26,
    "latitude": 37.85,
    "housing_median_age": 50,
    "total_rooms": 1120,
    "total_bedrooms": 283,
    "population": 697,
    "households": 264,
    "median_income": 2.125,
    "ocean_proximity": "NEAR BAY"
}
mydata = pd.DataFrame([mydata])
print(mydata)

mydata=Model.onehot_encoder(mydata)                                       # encode kar diya categorical columns ko

print("Prediction of New House: ",Model.predict(mydata)[0])


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














