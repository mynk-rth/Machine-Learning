import numpy as np
import matplotlib.pyplot as plt

class Bi_LinearRegression:
    def __init__(self,x,y):
        self.__x=np.array(x)
        self.__y=np.array(y)

    def coef(self):
        z=np.corrcoef(self.__x,self.__y)
        return z[0,1]
    
    def fit(self):
        mean_X = np.mean(self.__x)
        mean_Y = np.mean(self.__y)

        Xi_x = self.__x - mean_X

        Yi_y = self.__y - mean_Y

        Xi_x_sqr = Xi_x**2

        #                                                   ---------------- Slope (b1) --------------

        self.slope = sum(Xi_x * Yi_y) / sum(Xi_x_sqr)

        #                                                   ---------------- Y-Intercept (b0) --------------

        self.intercept = mean_Y - self.slope*mean_X


    def Predict(self,input):
        input_array = np.array(input)
        self.new_pred=self.intercept+self.slope*input_array
        return self.new_pred
        

    def MSE(self):
        diff = self.__y-self.Predict(self.__x)
        sq=diff**2
        mse=sq.sum()
        N=len(self.__x)

        plt.scatter(self.__x,diff)
        plt.axhline(0)
        plt.title("Residual Chart")
        plt.show()
        sq=diff**2

        # # Prediction for original data
        # new_pred = self.Predict(self.__x)
        # mse = np.mean((self.__y-new_pred) ** 2)
        # return mse

        return mse/N

    def drawRegressionLine(self):
        plt.scatter(self.__x,self.__y)
        plt.plot(self.__x,self.Predict(self.__x),label="Regression Line")
        plt.xlabel("Amount")
        plt.ylabel("Tip")
        plt.title("Regession line")
        plt.show()
