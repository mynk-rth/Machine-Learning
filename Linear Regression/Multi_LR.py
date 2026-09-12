import numpy as np
import matplotlib.pyplot as plt

class Multi_LinearRegression:
    def __init__(self,x1,x2,y):
        self.__x1=np.array(x1)
        self.__x2=np.array(x2)
        self.__y=np.array(y)

    def fit(self):

        #                                                   ---------------- Calculations --------------
        X = np.column_stack([np.ones(len(self.__x1)), self.__x1, self.__x2]).astype(int)
        X_transpose = X.T
        XTX_inv = np.linalg.inv(X_transpose @ X)
        A_cap = (XTX_inv @ X_transpose) @ self.__y

        #                            ---------------- Intercept(a0) & Slope(a1, a2 ,a3,..) --------------
        self.a0 = A_cap[0]
        self.a1 = A_cap[1]
        self.a2 = A_cap[2]

    def Predict(self,input1,input2):

        input_x1 = np.array(input1)
        input_x2 = np.array(input2)
        self.new_pred = self.a0 + self.a1* input_x1 + self.a2* input_x2
        return self.new_pred


    def MSE(self):
        diff = self.__y-self.Predict(self.__x1,self.__x2)
        sq=diff**2
        mse=sq.sum()
        N=len(self.__x1)

        # plt.figure(figsize=(10, 6))
        # plt.subplot(1,2,1)
        # plt.scatter(self.__x1,diff)
        # plt.axhline(0)
        # plt.title("Residual Chart x1 vs diff")

        # plt.subplot(1,2,2)
        # plt.scatter(self.__x2,diff)
        # plt.axhline(0)
        # plt.title("Residual Chart x2 vs diff")
        # plt.tight_layout()
        # plt.show()

        return mse/N

    # def drawRegressionLine(self):

    #     plt.figure(figsize=(10, 6))
    #     plt.subplot(1,2,1)
    #     plt.scatter(self.__x1,self.__y)
    #     plt.plot(self.__x1,self.new_pred)
    #     plt.xlabel("Product 1 sales")
    #     plt.ylabel("Weekly sales")
      

    #     plt.subplot(1,2,2)
    #     plt.scatter(self.__x2,self.__y)
    #     plt.plot(self.__x2,self.new_pred)
    #     plt.xlabel("Product 2 sales")
    #     plt.ylabel("Weekly sales")
    #     plt.tight_layout()  
    #     plt.show()


    # ---------------- Regression Graph ----------------

    def regression_graph(self):

        # Prediction for original data
        y_pred = self.Predict(self.__x1, self.__x2)

        plt.scatter(
            self.__y,
            y_pred,
            label="Predicted"
        )

        # Perfect prediction line
        plt.plot(
            [self.__y.min(), self.__y.max()],
            [self.__y.min(), self.__y.max()],
            label="Perfect Prediction"
        )

        plt.xlabel("Actual Y")
        plt.ylabel("Predicted Y")
        plt.title("Multiple Linear Regression")

        plt.legend()
        plt.grid(True)
        plt.show()
        
    # ---------------- X1 vs Y Graph ----------------

    def graph_x1_y(self):

        plt.scatter(self.__x1, self.__y)

        # Regression line of X1 and Y
        slope, intercept = np.polyfit(self.__x1, self.__y, 1)

        y_line = intercept + slope * self.__x1

        plt.plot(self.__x1, y_line)

        plt.xlabel("X1")
        plt.ylabel("Y")
        plt.title("X1 vs Y Regression")

        plt.grid(True)
        plt.show()

    # ---------------- X2 vs Y Graph ----------------

    def graph_x2_y(self):

        plt.scatter(self.__x2, self.__y)

        # Regression line of X2 and Y
        slope, intercept = np.polyfit(self.__x2, self.__y, 1)

        y_line = intercept + slope * self.__x2

        plt.plot(self.__x2, y_line)

        plt.xlabel("X2")
        plt.ylabel("Y")
        plt.title("X2 vs Y Regression")

        plt.grid(True)
        plt.show()
