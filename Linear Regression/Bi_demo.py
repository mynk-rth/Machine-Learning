from Bi_LinearRegression import Bi_LinearRegression

amount = [34,108,64,88,99,51]
tip = [5,17,11,8,14,5]

model = Bi_LinearRegression(amount, tip)


print("Correlation:", model.coef())


model.fit()
print("Intercept:", model.intercept)
print("Slope:", model.slope)


# test = [70, 2000]
# p = model.Predict(test)
# print(model.new_pred)

# model.Predict(amount)               
# print(model.new_pred)


#                                                   ---------------- MSE --------------


print("MSE :", model.MSE())

model.drawRegressionLine()