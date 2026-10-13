data = [['Rainy', 'Cool', 'Weak'], ['Rainy', 'Cool', 'Strong'], ['Sunny', 'Cool', 'Weak'], ['Rainy', 'Hot', 'Weak']]
x = ["Sunny","Cool","Weak"]



for j in range(len(x)):
            count = sum(x[j]==row[j] for row in data)
            print(j,count)
