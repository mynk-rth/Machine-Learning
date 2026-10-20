import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ID3 stands for Iterative Dichotomiser 3.abs

data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain",
        "Rain", "Rain", "Overcast", "Sunny",
        "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"
    ],
    "Temperature": [
        "Hot", "Hot", "Hot", "Mild",
        "Cool", "Cool", "Cool", "Mild",
        "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],
    "Humidity": [
        "High", "High", "High", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],
    "Wind": [
        "Weak", "Strong", "Weak", "Weak",
        "Weak", "Strong", "Strong", "Weak",
        "Weak", "Weak", "Strong", "Strong",
        "Weak", "Strong"
    ],
    "Play": [
        "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "Yes",
        "Yes", "No"
    ]
}

df = pd.DataFrame(data)

X = df.drop("Play",axis=1)
Y = df["Play"]

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

print("Encoded Features : ")
print(X_encoded)

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

print("Encoded Labels : ")
print(Y_encoded)


#-------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=0
)
model.fit(X_encoded,Y_encoded)

# prediction
new_weather = pd.DataFrame([{
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "Normal",
    "Wind": "Weak"
}])

new_encoded = encoder.transform(new_weather)
prediction = model.predict(new_encoded)

# Convert numeric prediction back to Yes/No
result = label_encoder.inverse_transform(prediction)

print("Tennis Decision : ",result[0])


# plt.figure(figsize=(16,12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
)
plt.title("Tennis Game Decision Tree")
plt.show()

