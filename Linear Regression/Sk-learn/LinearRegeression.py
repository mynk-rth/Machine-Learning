import pandas as pd
import numpy as np
import MLLib
class LinearRegression:
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
        dtype=int)
        # Make exactly the same columns as training data
        data_encode = data_encode.reindex(
        columns=self.encoder_columns,
        fill_value=0)
        data_final = pd.concat(
        [data_num, data_encode],
        axis=1)


     print("\nPrediction Data:")
     print(data_final)

    # Convert to numpy
     values = data_final.iloc[0].to_numpy(dtype=float)

    # Start with intercept
     S = self.intercept

    # Apply slopes
     for i, v in enumerate(values):
        S = S + self.slope[i] * v

     print("\nPredicted Price:", S)

     return S    

'''
model=LinearRegression()
model.loadFile("M:/DISK M/Python~SS/Machine Learning/Linear Regression/Datasets/mumbai_housing.csv")
cat=['locality','city','property_type','furnished']
num = ["area","bedroom_num","bathroom_num","balcony_num","age","total_floors"]
model.dataCleaning(cat,num,'price')

model.fit()

mydata = {

    "area": 200,
    "bedroom_num": 3,
    "bathroom_num": 3,
    "balcony_num": 2,
    "age": 2,
    "total_floors": 1,
    "locality": "Andheri West",
    "city": "Mumbai",
    "property_type": "Apartment",
    "furnished": "Furnished",

}

model.predict(mydata)
'''

'''
model=LinearRegression()
model.loadFile("M:/DISK M/Python~SS/Machine Learning/Linear Regression/Datasets/housing.csv")
cat=["ocean_proximity"]
num= [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    
   
]
model.dataCleaning(cat,num,'median_house_value')

model.fit()

mydata = {
"longitude":-122.26, 
 "latitude":37.85, 
"housing_median_age":50, 
 "total_rooms": 1120,
"total_bedrooms":283,
"population":697, 
"households":264,
"median_income":2.125,
"ocean_proximity":'NEAR BAY'
}

model.predict(mydata)
'''