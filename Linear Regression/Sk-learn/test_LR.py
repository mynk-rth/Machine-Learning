from sklearn.linear_model import LinearRegression
import numpy as np

amount = np.array([34,108,64,88,99,51])
amount = amount.reshape(len(amount),1)

tip = np.array([5,17,11,8,14,5])
tip = tip.reshape(len(tip),1)
# print(amount,tip)

model = LinearRegression()
model.fit(amount,tip)
print(model.intercept_)               # intercept
print(model.coef_)                    # slope

k = model.predict([[200],[180]])
print(k)

