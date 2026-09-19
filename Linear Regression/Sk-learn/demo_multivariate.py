import pandas as pd
from Multivariate_class import Multivariate_LR

#---------------------------------------------- Data Fetching -------------------------------------------
data = pd.read_csv(f"M:/DISK M/Python~SS/Machine Learning/Linear Regression/Sk-learn/mumbai_housing.csv")
# print(data)
# print(data.columns)

#---------------------------------------------- Remove Unnecssary columns -------------------------------------------
data = data.drop(['title','price_per_sqft','city','latitude','longitude'],axis=1)


#---------------------------------------------- Model -------------------------------------------
Model = Multivariate_LR()

Data_X, Data_Y = Model.superencoder(data,"price")                            # Data , target_column(data_Y column)

x_train,x_test,y_train,y_test = Model.test_train_splits(Data_X,Data_Y,0.8)  

Model.fit(x_train,y_train)

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

print("Prediction : ",Model.predict(new_house))

print("MSE : ",Model.MSE(x_test,y_test))
















# categorical_columns=Model.find_cat_cols(new_house)            # new_house ke categorical columns find kiye 
# Model.encoder(new_house,categorical_columns)                  # encode kar diya categorical columns ko
# new_house = new_house[Data_X.columns]                         # columns ko sahi order me fix kiya just like A_cap
# X_new = np.column_stack([
#     np.ones(len(new_house)),                                  # matric create kiya 
#     new_house
# ])