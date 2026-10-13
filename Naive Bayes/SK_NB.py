from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

X = [
    ["Sunny", "Hot", "Weak"],
    ["Sunny", "Hot", "Strong"],
    ["Rainy", "Cool", "Weak"],
    ["Rainy", "Cool", "Strong"],
    ["Sunny", "Cool", "Weak"],
    ["Rainy", "Hot", "Weak"],
]

y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]


# Outlook 
e1 = LabelEncoder()
X1 = e1.fit_transform([row[0] for row in X])

# Temperature 
e2 = LabelEncoder()
X2 = e2.fit_transform([row[1] for row in X])

# wind 
e3 = LabelEncoder()
X3 = e3.fit_transform([row[2] for row in X])

X_encoded = list(zip(X1,X2,X3))
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
        e3.transform(["Weak"])[0]
    ]
]

prediction = model.predict(new_data)
print(prediction)
print("Prediction : ",Y_encoder.inverse_transform(prediction)[0])

probability = model.predict_proba(new_data)
print("Probability : ",probability)