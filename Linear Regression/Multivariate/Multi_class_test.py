from Multi_LR import Multi_LinearRegression

x1 = [1,2,3,4]
x2 = [4,5,8,2]
y = [1,6,8,12]

model = Multi_LinearRegression(x1, x2, y)

model.fit()
print("Intercept : ",model.a0)
print("Slope a1 : ",model.a1)
print("Slope a2 : ",model.a2)

# model.Predict(4,2)               
# print(model.new_pred)

model.Predict(x1,x2)               
print(model.new_pred)


#                                                   ---------------- MSE --------------

print("MSE :", model.MSE())


# model.drawRegressionLine()

model.regression_graph()

model.graph_x1_y()
model.graph_x2_y()