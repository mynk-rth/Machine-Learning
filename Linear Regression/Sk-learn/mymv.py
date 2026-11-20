import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv(f"M:/DISK M/Python Data Science AI & ML/Machine Learning/Linear Regression/Datasets/mumbai_housing.csv")

num = ["area", "bedroom_num", "bathroom_num", "balcony_num", "age", "total_floors"]
cat = ["locality", "property_type", "furnished"]
target = ["price"]

required_columns = (
    num
    + cat
    + target
)

df = df.dropna(subset=required_columns)

X = df[num + cat]
y = df["price"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
], remainder="passthrough")

X = pre.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)

print("Score:", model.score(X_test, y_test))

y_pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("MODEL PERFORMANCE")
print("MAE       :", mae)
print("MSE       :", mse)
print("R2 Score  :", r2)
print(f"R2 (%)    : {r2 * 100:.2f}%")

# SHOW ACTUAL VS PREDICTED
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")

print(result.head(20))


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

prediction = model.predict(
    pre.transform(new_house[num + cat])
)

print("Predicted Price:", prediction[0])