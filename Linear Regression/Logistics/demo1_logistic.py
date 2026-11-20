import pandas as pd
import numpy as np
from Logistic_class import Logistic_Regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#---------------------------------------------- Data Fetching -------------------------------------------
data = pd.read_csv(f"M:/DISK M/Python Data Science AI & ML/Machine Learning/Linear Regression/Datasets/creditScore.csv")
# print(data)
# print(data.columns)


#---------------------------------------------- Model -------------------------------------------
Model = Logistic_Regression()

Data_X, Data_Y = Model.datapreprocessing(data,"approved")                            # Data , target_column(data_Y column)

x_train,x_test,y_train,y_test = Model.test_train_splits(Data_X,Data_Y,1)  

Model.fit(x_train,y_train)

new_input = pd.DataFrame({
    "creditScore": [500]
})

new_prediction = Model.predict(new_input)

print("New prediction : ", new_prediction)
print("Probability : ", Model.estimated_probability.tolist())


# ---------------------------------------------- MSE -------------------------------------------

y_pred = Model.predict(x_train)
# print("Prediction of x_train : \n",y_pred)


print("MSE : ",Model.MSE(y_train,y_pred))


# SHOW ACTUAL VS PREDICTED
result = pd.DataFrame({
    "Actual": y_train,
    "Predicted": y_pred
})
print("\nActual vs Predicted:")
print(result)


# ---------------------------------------------- Model Performance (sklearn) -------------------------------------------
accuracy = accuracy_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)
class_report = classification_report(y_train, y_pred, zero_division=0)

print("\n---------------- Model Performance ----------------")
print(f"Accuracy Score : {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)














