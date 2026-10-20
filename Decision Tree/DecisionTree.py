import pandas as pd
import math

df=pd.DataFrame({'outlook':['sunny','sunny','overcast','rainy','rainy','rainy','overcast','sunny','sunny','rainy','sunny','overcast','overcast','rainy'],
                'temperature':['hot','hot','hot','mild','cool','cool','cool','mild','cool','mild','mild','mild','hot','mild'],
                'humidity':['high','high','high','high','normal','normal','normal','high','normal','normal','normal','high','normal','high'],
                'wind':['weak','strong','weak','weak','weak','strong','strong','weak','weak','weak','strong','strong','weak','strong'],
                'play':['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']})

# Entropy ka formula


def entropy(data):

    total = len(data)
    print(total)

    
    yes = data.count("yes")
    no = data.count("no")   

    p1 = yes / total
    p2 = no / total

    result = 0
    if p1 !=0:
        result = result - (p1 * math.log2(p1))
        
    if p2 !=0:
        result = result - (p2 * math.log2(p2))
        
    return result

   # return -(p1 * math.log2(p1) + p2 * math.log2(p2))


# Overall Entropy
E = entropy(9, 5)

print("Entropy =", E)


# Outlook ke according entropy
sunny = entropy(2, 3)
overcast = entropy(4, 0)
rain = entropy(3, 2)

print("Sunny =", sunny)
print("Overcast =", overcast)
print("Rain =", rain)


# Information Gain
gain = E - ((5/14) * sunny + (4/14) * overcast + (5/14) * rain )

print("Information Gain =", gain)

#--------------------Nearest bias-----------------------------

# class NB:
#     def fit (self,x,y):
#         self.classes=list(set(y))
#         print("classes=", self.classes)
# X= [
#     [" Sunny", "Hot", "Weak"],
#     ["Sunny", "Hot", "Strong"],
#     ["Rainy", "Cool", "Weak"],
#     ["Rainy", "Cool", "Strong"],
#     ["Sunny", "Cool", "Weak"],
#     ["Rainy", "Hot", "Weak"]
# ]     

# Y = ["No", "No", "Yes", "Yes", "Yes", "Yes"] 

# model = NB()
# model.fit(X,Y)