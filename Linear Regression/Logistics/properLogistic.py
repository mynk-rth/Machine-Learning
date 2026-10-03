# Proper Logistic Regression ke liye fit() bhi change karo
# Tumhari existing coding style ko maximum preserve karte hue:


def fit(self, x_train, y_train):

    X = x_train.copy()

    X.insert(0, "intercept", 1)

    X = X.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)

    # Initially coefficients = 0
    self.A_cap = np.zeros(X.shape[1])

    learning_rate = 0.01
    epochs = 1000

    for i in range(epochs):

        # Linear equation
        z = X @ self.A_cap

        # Sigmoid
        probability = 1 / (1 + np.exp(-z))

        # Error
        error = probability - y

        # Gradient
        gradient = (X.T @ error) / len(X)

        # Update coefficients
        self.A_cap = self.A_cap - learning_rate * gradient

    print("\n\t\tModel Training Completed\n")



# Ab tumhara complete flow actual Logistic Regression ke close aa jayega:
        #          FIT
        #           ↓
        #     X + intercept
        #           ↓
        #   Initial A_cap = 0
        #           ↓
        #  X @ A_cap → z
        #           ↓
        #      Sigmoid
        #           ↓
        #    probability
        #           ↓
        # error = probability-y
        #           ↓
        #      gradient
        #           ↓
        #   update A_cap
        #           ↓
        #     repeat epochs

# Phir predict():
# New Input
#    ↓
# X @ learned A_cap
#    ↓
# z
#    ↓
# Sigmoid
#    ↓
# Probability
#    ↓
# Probability >= 0.5
#    ↓
# 0 / 1