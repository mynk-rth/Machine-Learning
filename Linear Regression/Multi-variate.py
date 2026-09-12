import numpy as np

x1 = np.array([1,2,3,4])
x2 = np.array([4,5,8,2])
y = np.array([1,6,8,12])

# print("x1 : ",x1)
# print("x2 : ",x2)
# print("y : ",y)


#                                                   ---------------- Calculations --------------

X = np.column_stack([np.ones(len(x1)), x1, x2]).astype(int)
# print(X)


X_transpose = X.T
# print(X_transpose)


XTX_inv = np.linalg.inv(X_transpose @ X)
# print(XTX_inv)


A_cap = (XTX_inv @ X_transpose) @ y
print("A_cap : ",A_cap) 


#                                                   ---------------- Intercept(a0) & Slope(a1, a2 ,a3,..) --------------

a0 = A_cap[0]
a1 = A_cap[1]
a2 = A_cap[2]

print("Intercept : ",a0)
print("Slope a1 : ",a1)
print("Slope a2 : ",a2)


#                                                   ---------------- Regression Line --------------

input_x1 = int(input("Enter Product 1 Sales : "))
input_x2 = int(input("Enter Product 2 Sales : "))

weekly_sales = a0 + a1* input_x1 + a2* input_x2
print(f"Predict Weekly Sales : {weekly_sales}")