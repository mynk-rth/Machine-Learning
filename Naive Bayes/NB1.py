class NB:
    def __init__(self):
        pass

    def fit(self,X,Y):
        self.X = X
        self.Y = Y
        self.classes = list(set(self.Y))          #p(c)
        print(self.classes)

        self.data = {}
        for c in self.classes:
            self.data[c] = [ X[i] for i in range(len(Y)) if c==Y[i]]
        print(self.data)

        self.prob = {}
        for c in self.classes:
            self.prob[c] = len(self.data[c])/len(Y)
        print(self.prob)


    def predict(self,x):
        
        result = {}
        for c in self.classes:
            rows = self.data[c]
            p = self.prob[c]
            for j in range(len(x)):
                count = sum(x[j]==row[j] for row in rows)
                values = len(set(row[j] for row in self.X))
                p *= (count+1)/(len(rows)+values)
            result[c] = p
        print("Evidence : ")
        print(result)

        evidence = sum(result.values())
        for r in result:
            result[r] = result[r]/evidence
        print(result)
        print(max(result,key=result.get))








X = [
    ["Sunny", "Hot", "Weak"],
    ["Sunny", "Hot", "Strong"],
    ["Rainy", "Cool", "Weak"],
    ["Rainy", "Cool", "Strong"],
    ["Sunny", "Cool", "Weak"],
    ["Rainy", "Hot", "Weak"],
]

y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]


model = NB()
model.fit(X,y)

model.predict(["Sunny","Cool","Weak"])