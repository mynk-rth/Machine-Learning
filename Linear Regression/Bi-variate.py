import numpy as np

x = np.array([34,108,64,88,99,51])                   # Total Bill ($)
y = np.array([5,17,11,8,14,5])                       # Tip Amount ($)



#                                                 ------------ Coefficent of Correlation --------------


coef_corr=np.corrcoef(x,y)
print(f"Coefficent of Correlation = {coef_corr[0,1]}")


#                                                   ---------------- Calculations --------------


sum_X = sum(x)
sum_Y = sum(y)
# print(sum_X)
# print(sum_Y)


mean_X = np.mean(x)
mean_Y = np.mean(y)
# print(mean_X)
# print(mean_Y)


Bill_deviation = x - mean_X
# print(Bill_deviation)


Tip_deviation = y - mean_Y
# print(Tip_deviation)


Deviation_Products = Bill_deviation * Tip_deviation
# print(Deviation_Products)


Bill_deviation_sqr = Bill_deviation**2
# print(Bill_deviation_sqr)


#                                                   ---------------- Slope (b1) --------------

slope = sum(Bill_deviation * Tip_deviation) / sum(Bill_deviation_sqr)
print(f"Slope = {slope}")


#                                                   ---------------- Y-Intercept (b0) --------------

Y_intercept = mean_Y - slope*mean_X
print(f"Y-Intercept = {Y_intercept} \n\n")


#                                                   ---------------- Regression Line --------------

Bill = int(input("Enter Bill Amount : "))
tip = Y_intercept + slope * x
print(f"Predict Tip Amount : {tip}")


#                                                   ---------------- MSE  --------------


