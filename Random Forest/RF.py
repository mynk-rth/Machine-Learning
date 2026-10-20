import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Bank Loan Dataset
data = {
    'Income': [
        25000, 30000, 40000, 60000, 20000,
        80000, 45000, 90000, 35000, 70000,
        28000, 65000, 55000, 40000, 100000
    ],
    'CreditScore': [
        550, 600, 650, 800, 500,
        850, 700, 900, 620, 820,
        580, 780, 740, 650, 920
    ],
    'LoanAmount': [
        200000, 250000, 240000, 350000, 180000,
        400000, 250000, 450000, 220000, 300000,
        200000, 280000, 320000, 240000, 500000
    ],
    'Experience': [
        1, 2, 3, 7, 1, 10, 4, 12, 2, 8, 1, 6, 5, 3, 15
    ],
    'LoanStatus': [
        'Reject', 'Reject', 'Reject', 'Approve', 'Reject',
        'Approve', 'Approve', 'Approve', 'Reject', 'Approve',
        'Reject', 'Approve', 'Approve', 'Reject', 'Approve'
    ]
}

df = pd.DataFrame(data)

print("Bank Loan Dataset:")
print(df)

# Input and Output
X = df[['Income', 'CreditScore', 'LoanAmount', 'Experience']]
Y = df['LoanStatus']

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, Y_train)

# Prediction on Test Data
y_pred = model.predict(X_test)

print("\nActual Results:")
print(Y_test.tolist())

print("\nPredicted Results:")
print(y_pred.tolist())

# Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("\nAccuracy:", accuracy)

# New Customer
new_customer = pd.DataFrame({
    'Income': [75000],
    'CreditScore': [820],
    'LoanAmount': [300000],
    'Experience': [7]
})

prediction = model.predict(new_customer)

print("\nNew Customer Loan Status:", prediction[0])