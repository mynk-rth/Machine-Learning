import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
## Simple Linear Regression Implementation
class LinearRegression:
    def __init__(self,x,y):
        self.__x=np.array(x)
        self.__y=np.array(y)
    def CC(self):
        x=np.array([34,108,64,88,99,51])
        y=np.array([5,17,11,8,14,5])
        sx=sum(self.__x)
        sy=sum(self.__y)
        sxy=sum(self.__x*self.__y)
        xx=self.__x*self.__x
        yy=self.__y*self.__y
        sxx=sum(xx)
        syy=sum(yy)
        N=len(self.__x)
        r=(N*sxy-sx*sy)/np.sqrt((N*sxx-sx**2)*(N*syy-sy**2))
        return r
    def fit(self):
        self.__slope=sum((self.__x-np.mean(self.__x))*(self.__y-np.mean(self.__y)))/sum((self.__x-np.mean(self.__x))**2)
        self.__intercept=np.mean(self.__y)-self.__slope*np.mean(self.__x)
        return self.__slope, self.__intercept
    def predict(self,input):
        R=[]
        for V in input:
            r=self.__intercept+self.__slope*V
            R.append(r)
        return np.array(R)
    def MSE(self):
        y_pred=self.__intercept+self.__slope*self.__x
        mse=sum((self.__y-y_pred)**2)/len(self.__y)
        return mse



## Multivatriate Linear Regression Implementation
class MultipleLinearRegression:
    def __init__(self, Product1, Product2, Weekly_Sales):

        self.__Product1 = np.array(Product1)
        self.__Product2 = np.array(Product2)
        self.__Weekly_Sales = np.array(Weekly_Sales)

        self.__a0 = 0
        self.__a1 = 0
        self.__a2 = 0

    # Function 1: Check Training Data Length
    def check_training_length(self):
        if len(self.__Product1) == len(self.__Product2) and len(self.__Product2) == len(self.__Weekly_Sales):
            return True
        else:
            return False

    # Function 2: Create X Matrix
    def create_x_matrix(self):
        self.__X = np.column_stack((np.ones(len(self.__Product1)),self.__Product1,self.__Product2))
        return self.__X

    # Function 3: Calculate Coefficients
    def calculate_coefficients(self):

        self.__A = (np.linalg.inv(self.__X.T @ self.__X) @ self.__X.T) @ self.__Weekly_Sales
        self.__a0 = self.__A[0]
        self.__a1 = self.__A[1]
        self.__a2 = self.__A[2]
        return self.__A

    # Graph:1
    #Function 4: Graph for Product1 vs weekly sales
    def product1_vs_weekly_sales(self):
        fig,ax=plt.subplots()
        ax.scatter(self.__Product1,self.__Weekly_Sales)
        ax.set_xlabel("Product1",fontsize=12)
        ax.set_ylabel("Weekly Sales",fontsize=12)
        ax.set_title("Product1 VS Weekly Sales",fontsize=14)
        return fig

    # Graph:2
    # Function 5: Graph for Product1 vs weekly sales 
    def product1_with_line(self):
        fig,ax=plt.subplots()
        ax.scatter(self.__Product1,self.__Weekly_Sales,label="Actual Data")
        # Product1 ke according line
        x_line = np.linspace(self.__Product1.min(),self.__Product1.max(),100)
        # Sirf Product1 ka effect
        y_line = self.__a0 + self.__a1 * x_line
        ax.plot(x_line, y_line, label="Regression Line")
        ax.set_xlabel("Product1",fontsize=12)
        ax.set_ylabel("Weekly Sales",fontsize=12)
        ax.set_title("Product1 vs Weekly Sales with Regression Line",fontsize=14)
        ax.legend()
        return fig

    # Graph:3
    # Function 6: Graph for Product2 vs weekly sales
    def product2_vs_weekly_sales(self):
        fig,ax=plt.subplots()
        ax.scatter(self.__Product2,self.__Weekly_Sales)
        ax.set_xlabel("Product2",fontsize=12)
        ax.set_ylabel("Weekly Sales",fontsize=12)
        ax.set_title("Product2 vs Weekly Sales",fontsize=14)
        return fig

    # Graph:4
    # Function 7: Graph for Product2 vs weekly sales with regresssion line
    def product2_with_line(self):
        fig,ax=plt.subplots()
        ax.scatter(self.__Product2,self.__Weekly_Sales,label="Actual Data")
        # Product2 ke according line
        x_line = np.linspace(self.__Product2.min(),self.__Product2.max(),100)
        # Sirf Product2 ka effect
        y_line = self.__a0 + self.__a2 * x_line
        ax.plot(x_line, y_line, label="Regression Line")
        ax.set_xlabel("Product2",fontsize=12)
        ax.set_ylabel("Weekly Sales",fontsize=12)
        ax.set_title("Product2 vs Weekly Sales with Regression Line",fontsize=14)
        ax.legend()
        return fig

    # Function 8: Take Input for Prediction and prediction
    def predict(self,X1,X2):
        self.__X1 = np.array(X1)
        self.__X2 = np.array(X2)
        self.__Y_pred = (self.__a0+ self.__a1 * self.__X1+ self.__a2 * self.__X2)
        return self.__Y_pred

    # GRAPH 5
    # Function 9: Graph for Predicted Weekly Sales
    def prediction_graph(self):
        fig, ax = plt.subplots()
        x = np.arange(1, len(self.__Y_pred) + 1)
        ax.plot(x,self.__Y_pred,marker="o",label="Predicted Weekly Sales")
        ax.set_xlabel("Prediction Number", fontsize=12)
        ax.set_ylabel("Predicted Weekly Sales", fontsize=12)
        ax.set_title("Predicted Weekly Sales", fontsize=14)
        ax.legend()
        return fig
    # # Function 9: Actual vs Predicted
    # def actual_vs_predicted(self):
    #     fig,ax=plt.subplots()
    #     ax.scatter(self.__Weekly_Sales,self.__Y_pred,label="Predicted VS Actual")
    #     # Perfect prediction line
    #     minimum = min( self.__Weekly_Sales.min(),self.__Y_pred.min())
    #     maximum = max(self.__Weekly_Sales.max(),self.__Y_pred.max())

    #     ax.plot([minimum, maximum],[minimum, maximum],label="Perfect Prediction")
    #     ax.set_xlabel("Actual Weekly Sales",fontsize=12)
    #     ax.set_ylabel("Predicted Weekly Sales",fontsize=12)
    #     ax.set_title("Actual vs Predicted Weekly Sales",fontsize=14)
    #     ax.legend()
    #     return fig


