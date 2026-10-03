import pandas as pd
import numpy as np
import MLLib
import math
class LogisticRegression:
    def loadFile(self,filename):
        self.data=pd.read_csv(filename)
        self.data=self.data.dropna()
        self.actual_data=self.data.copy()
    def dataCleaning(self,categorial,numerical,target):
        self.categorial=categorial
        self.numerical=numerical
        self.target=target
        if(len(categorial)>0):
          data_encode,self.encoder_columns =MLLib.encoder(self.data,categorial)
          self.data=pd.concat([self.data[numerical],data_encode],axis=1)
          self.columns=numerical+list(self.encoder_columns)

        else:
           self.data=self.data[numerical]  
           self.columns=numerical
        
       
        

    def fit(self):
        X=self.data
        Y=self.actual_data[self.target]
        X_train,Y_train,X_test,Y_test=MLLib.test_train_splits(X,Y)
        print(len(Y_train),len(X_train))
        X_train=X_train.to_numpy()
        X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
        Y_train=Y_train.to_numpy()
        result=np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train
        self.intercept=result[0]
        self.slope=result[1:]
        print(result)
        print("xxxx",len(result))
          

        for i,v in enumerate(self.columns):
            print(v,self.slope[i]) 
    def predict(self, mydata):

     # Convert dictionary to DataFrame
     data = pd.DataFrame([mydata])
     data_num = data[self.numerical]
     if(len(self.categorial)>0):
       # Encode categorical columns
        data_encode = pd.get_dummies(
        data[self.categorial],
        columns=self.categorial,
        dtype=int
       )

        # Make exactly the same columns as training data
        data_encode = data_encode.reindex(
        columns=self.encoder_columns,
        fill_value=0
        )

        if(len(data_encode)>0):
          data_final = pd.concat(
          [data_num, data_encode],
          axis=1
    )
     else:
       data_final=data_num


     print("\nPrediction Data:")
     print(data_final)

    # Convert to numpy
     values = data_final.iloc[0].to_numpy(dtype=float)

    # Start with intercept
     S = self.intercept

    # Apply slopes
     for i, v in enumerate(values):
        S = S + self.slope[i] * v
  
     r=math.exp(S)/(1+math.exp(S))
     print("\nPredicted Price:", r)

     return r




model=LogisticRegression()
model.loadFile("M:/DISK M/Python~SS/Machine Learning/Linear Regression/Datasets/heart.csv")
cat=[]


num= [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",  
]
model.dataCleaning(cat,num,'target')

model.fit()

mydata = {
"age":50,
    "sex":0,
    "cp":2,
    "trestbps":120,
    "chol":219,
    "fbs":0,
    "restecg":1,
    "thalach":158,
    "exang":0,
    "oldpeak":1.6,
    "slope":1,
    "ca":0,
    "thal":2,
    
}

model.predict(mydata)
