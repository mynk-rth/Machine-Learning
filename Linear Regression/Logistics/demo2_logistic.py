import pandas as pd
import numpy as np
from Logistic_class import Logistic_Regression


#---------------------------------------------- Data Fetching -------------------------------------------
data = pd.read_csv(f"M:/DISK M/Python~SS/Machine Learning/Linear Regression/Datasets/heart_failure_clinical_records_dataset.csv")
# print(data)
# print(data.columns)


#---------------------------------------------- Model -------------------------------------------
Model = Logistic_Regression()

Data_X, Data_Y = Model.datapreprocessing(data,"DEATH_EVENT")                            # Data , target_column(data_Y column)

x_train,x_test,y_train,y_test = Model.test_train_splits(Data_X,Data_Y,1)  

Model.fit(x_train,y_train)

new_input = pd.DataFrame({
    "age": [50.0],
    "anaemia": [0],
    "creatinine_phosphokinase": [196],
    "diabetes": [0],
    "ejection_fraction": [45],
    "high_blood_pressure": [0],
    "platelets": [395000.00],
    "serum_creatinine": [1.6],
    "serum_sodium": [136],
    "sex": [1],
    "smoking": [1],
    "time": [285]
})

new_prediction = Model.predict(new_input)

print("New prediction : ", new_prediction)
print("Probability : ", Model.estimated_probability.tolist())


# ---------------------------------------------- MSE -------------------------------------------

y_pred = Model.predict(x_train)
# print("Prediction of x_test : \n",y_pred)
# print(type(y_pred))


print("MSE : ",Model.MSE(y_train,y_pred))

# SHOW ACTUAL VS PREDICTED

result = pd.DataFrame({
    "Actual": y_train,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")

print(result.tail(10))














