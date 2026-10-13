from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

X = [
    ["Sunny", "Hot", "High", "Weak"],
    ["Sunny", "Hot", "High", "Strong"],
    ["Overcast", "Hot", "High", "Weak"],
    ["Rain", "Mild", "High", "Weak"],
    ["Rain", "Cool", "Normal", "Weak"],
    ["Rain", "Cool", "Normal", "Strong"],
    ["Overcast", "Cool", "Normal", "Strong"],
    ["Sunny", "Mild", "High", "Weak"],
    ["Sunny", "Cool", "Normal", "Weak"],
    ["Rain", "Mild", "Normal", "Weak"],
    ["Sunny", "Mild", "Normal", "Strong"],
    ["Overcast", "Mild", "High", "Strong"],
    ["Overcast", "Hot", "Normal", "Weak"],
    ["Rain", "Mild", "High", "Strong"]
]

y = [
    "No", "No", "Yes", "Yes",
    "Yes", "No", "Yes", "No",
    "Yes", "Yes", "Yes", "Yes",
    "Yes", "No"
]

# Outlook 
e1 = LabelEncoder()
X1 = e1.fit_transform([row[0] for row in X])

# Temperature 
e2 = LabelEncoder()
X2 = e2.fit_transform([row[1] for row in X])

# Humidity 
e3 = LabelEncoder()
X3 = e3.fit_transform([row[2] for row in X])

# Wind 
e4 = LabelEncoder()
X4 = e4.fit_transform([row[3] for row in X])

X_encoded = list(zip(X1,X2,X3,X4))
print("Encoded X :")
print(X_encoded)

Y_encoder = LabelEncoder()
Y_encoded = Y_encoder.fit_transform(y)



model = CategoricalNB()
model.fit(X_encoded,Y_encoded)

new_data = [
    [
        e1.transform(["Sunny"])[0],
        e2.transform(["Cool"])[0],
        e3.transform(["High"])[0],
        e4.transform(["Weak"])[0]
    ]
]



prediction = model.predict(new_data)
print(prediction)
print("Prediction : ",Y_encoder.inverse_transform(prediction)[0])

probability = model.predict_proba(new_data)
print("Probability : ",probability)