import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



file_path = "F:/data/mumbai.csv"

df = pd.read_csv(file_path)




#missing valuessss


print("\nMissing Values:")
print(df.isnull().sum())



# features alag karlo


# numeric
numerical_columns = [
    "area",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "age",
    "total_floors"
]

# categorical
categorical_columns = [
    "locality",
    "property_type",
    "furnished"
]

# target
target_column = "price"



# romve missssing values


required_columns = (
    numerical_columns
    + categorical_columns
    + [target_column]
)

df = df.dropna(subset=required_columns)

print("\nShape after removing missing values:")
print(df.shape)



# x aur y data set karlo
X = df[
    numerical_columns + categorical_columns
]

y = df[target_column]



# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))



#data encoding jo humne get_dummies se kiya tha

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# Fit encoder only on training data
X_train_cat = encoder.fit_transform(
    X_train[categorical_columns]
)

# Transform test data
X_test_cat = encoder.transform(
    X_test[categorical_columns]
)



# GET ENCODED COLUMN NAMES


encoded_columns = encoder.get_feature_names_out(
    categorical_columns
)

print("\nNumber of categorical features:",
      len(encoded_columns))



#CREATE DATAFRAME FOR ENCODED DATA


X_train_cat = pd.DataFrame(
    X_train_cat,
    columns=encoded_columns,
    index=X_train.index
)

X_test_cat = pd.DataFrame(
    X_test_cat,
    columns=encoded_columns,
    index=X_test.index
)



# GET NUMERICAL DATA


X_train_num = X_train[numerical_columns]

X_test_num = X_test[numerical_columns]



# COMBINE NUMERICAL + CATEGORICAL


X_train_final = pd.concat(
    [
        X_train_num,
        X_train_cat
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_num,
        X_test_cat
    ],
    axis=1
)


print("\nFinal Training Shape:")
print(X_train_final.shape)

print("\nFinal Testing Shape:")
print(X_test_final.shape)



#LINEAR REGRESSION MODEL


model = LinearRegression()


# ==========================================================
# 13. TRAIN MODEL
# ==========================================================

model.fit(
    X_train_final,
    y_train
)

print("\nModel Training Completed")



#PREDICTION


y_pred = model.predict(
    X_test_final
)



# MODEL EVALUATION


mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)





print("MODEL PERFORMANCE")


print("MAE  :", mae)

print("MSE  :", mse)



# SHOW ACTUAL VS PREDICTED


result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")

print(result.head(20))



# PREDICT NEW HOUSE


new_house = pd.DataFrame({

    "area": [1200],

    "bedroom_num": [2],

    "bathroom_num": [2],

    "balcony_num": [1],

    "age": [5],

    "total_floors": [10],

    "locality": ["Andheri West"],

    "property_type": ["Apartment"],

    "furnished": ["Furnished"]
})



#ENCODE NEW HOUSE


new_house_cat = encoder.transform(
    new_house[categorical_columns]
)

new_house_cat = pd.DataFrame(
    new_house_cat,
    columns=encoded_columns
)



# NUMERICAL DATA OF NEW HOUSE


new_house_num = new_house[
    numerical_columns
].reset_index(drop=True)



# COMBINE NEW HOUSE DATA


new_house_final = pd.concat(
    [
        new_house_num,
        new_house_cat
    ],
    axis=1
)



#PREDICT PRICE


prediction = model.predict(
    new_house_final
)



print("NEW HOUSE PREDICTION")


print("Area:", new_house["area"].iloc[0])

print(
    "Bedrooms:",
    new_house["bedroom_num"].iloc[0]
)

print(
    "Bathrooms:",
    new_house["bathroom_num"].iloc[0]
)

print(
    "Locality:",
    new_house["locality"].iloc[0]
)

print(
    "Furnished:",
    new_house["furnished"].iloc[0]
)

print(
    "\nPredicted Price:",
    prediction[0]
)