import numpy as np 
import matplotlib.pyplot as plt

class SVM:

    def __init__(self):
        self.w = np.zeros(2)       # weight
        self.b = 0                  # bais

    def fit(self,X,Y, lr=0.01,epoch=10):
        # The learing rate controls how big a step the model takes when updating its weights and bais.
    #   for j in range(epoch):
        for i in range(len(X)):

            score = np.dot(self.w, X[i])+ self.b
            # Negative for cat/Positive for dog/0is hyperplane

            # Check SVM margin (for mis/classfication)
            '''
            Remember these three points:

            score = 0 -> Hyperplane

            y[i] * score >= 1 -> Margin requirement satisfied

            y[i] * score < 1 -> Update weighhts and bias in our simplified algorithm
            '''

            if Y[i] * score<1 : 
                self.w = self.w + lr * Y[i] * X[i]
                self.b = self.b + lr * Y[i]

    def predict(self, X):

        score = np.dot(self.w,X) + self.b

        if score>0:
            return "Dog"
        else:
            return "Cat"


    def plot(self, X, y):

        # Plot Cats
        plt.scatter(
        X[y == -1, 0], X[y == -1, 1],
        color="blue", label="Cat"
        )

        # Plot Dogs
        plt.scatter(
            X[y == 1, 0], X[y == 1, 1],
            color="red", marker="s", label="Dog"
        )

        plt.title("SVM: Cat vs Dog")
        plt.xlim(0, 30)
        plt.ylim(0, 75)
        plt.legend()
        plt.grid(True)
        plt.show()



# Weight and height scaled between 0 and 1
X = np.array([
    [2, 20],    # Cat
    [3, 25],    # Cat
    [4, 30],    # Cat
    [15, 50],   # Dog
    [20, 60],   # Dog
    [25, 70]    # Dog
])

# Cat = -1, Dog = +1
y = np.array([-1, -1, -1, 1, 1, 1])

model = SVM()
model.fit(X,y)

# Test a new animal
new_animal = np.array([5,25])

print("Prediction : ",model.predict(new_animal))
print("Weights : ",model.w)
print("Bais : ",model.b)
model.plot(X,y)