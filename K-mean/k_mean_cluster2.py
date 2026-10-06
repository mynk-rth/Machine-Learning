import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
class Kmean:
    def __init__(self,k):
        self.k=k
        self.centroid=None
        self.clusters=None

    def fit(self,data,centroid):
        #Initial centroids
        self.centroid=centroid
        while(True):
            #Har iteration mein clusters empty hoga
            clusters=[[] for i in range(self.k)]
            #Har data point ko nearest centroid do
            for v in data:
                distance=[]
                for c in self.centroid:
                    dis=np.sum(abs(v-np.array(c)))
                    distance.append(dis)
                cluster_index=distance.index(min(distance))
                clusters[cluster_index].append(v)
            #Purane centroid save karo
            old_centroid=self.centroid.copy()
            #New centroid calculate karo
            new_centroid=[]
            for i,cluster in enumerate(clusters):
                if len(cluster)>0:
                    mean=np.mean(cluster,axis=0)
                    new_centroid.append(mean.tolist())
                else:
                    new_centroid.append(old_centroid[i])
            #Centroid update karo
            self.centroid=new_centroid
            #Check centroid change hua ya nahi
            if old_centroid==self.centroid:
                break
        #Final clusters 
        self.clusters=[[v.tolist() for v in cluster] for cluster in clusters]
        print("Final Centroids:")
        print(self.centroid)
        print("\nFinal Clusters:")
        print(self.clusters)

    def graph(self):
        for i,cluster in enumerate(self.clusters):
            x=[]
            y=[]
            for point in cluster:
                x.append(point[0])
                y.append(point[1])
            plt.scatter(x,y,label=f"cluster{i+1}")
            #show centroids
        for i,centroid in enumerate(self.centroid):
            plt.scatter(centroid[0],centroid[1],marker="+",s=150,label=f"Centroid {i+1}")
        plt.xlabel("Study Hours")
        plt.ylabel("Attendence")
        plt.title("K-Menas Clusters")
        plt.legend()
        plt.grid(True)
        plt.show()



df=pd.DataFrame({'Student':['A','B','C','D','E','F'],
                 'Study Hours':[2,3,4,8,9,10],
                 'Attendence':[50,55,60,85,90,95],
                 'Marks':[45,50,55,80,88,92]})
data=df.iloc[:,1:].to_numpy()
print(df)
print(data)
#MODEL
centroid=ast.literal_eval(input("Enter the initial centroid: "))
model=Kmean(len(centroid))
model.fit(data,centroid)
model.graph()