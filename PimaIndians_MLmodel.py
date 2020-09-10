# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load the dataset
df_pima=pd.read_csv('C:/Users/MADHURA/Desktop/Pima_Project/PimaIndian_diabetes_Dataset.csv')
df_pima.head(20)

df_pima.info()

# Perform Data Pre-processing

df_pima.isnull().sum()

df_pima.head(20)


# Replacing 0's with NaNs

df_pima['Glucose'] = df_pima['Glucose'].replace('0', np.nan)
df_pima['BloodPressure'] = df_pima['BloodPressure'].replace('0', np.nan) 
df_pima['SkinThickness'] = df_pima['SkinThickness'].replace('0', np.nan) 
df_pima['Insulin'] = df_pima['Insulin'].replace('0', np.nan)        
df_pima['BMI'] = df_pima['BMI'].replace('0', np.nan) 
df_pima['DiabetesPedigreeFunction'] = df_pima['DiabetesPedigreeFunction'].replace('0', np.nan) 
df_pima['Age'] = df_pima['Age'].replace('0', np.nan)

df_pima.head(10)

df_pima.info()

# Replacing NaNs with mean value

df_pima['BMI'].fillna(df_pima['BMI'].median(), inplace=True)
df_pima['Glucose'].fillna(df_pima['Glucose'].median(), inplace=True)
df_pima['BloodPressure'].fillna(df_pima['BloodPressure'].median(), inplace=True)
df_pima['SkinThickness'].fillna(df_pima['SkinThickness'].median(), inplace=True)
df_pima['Insulin'].fillna(df_pima['Insulin'].median(), inplace=True)

df_pima.head()

# Selecting Features

X = pd.DataFrame(data = df_pima, columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]).values
y = pd.DataFrame(data = df_pima, columns = ["Outcome"]).values

# Fixing the random_state in order to maintain consistency in results
rand_state = 11

X_train, X_test, y_train, y_test = train_test_split(X , y, random_state=rand_state, test_size=0.2, stratify=y)
X_train.shape, X_test.shape
LR = LogisticRegression(random_state=rand_state, class_weight={0:0.33, 1:0.67})
LR.fit(X_train, y_train.ravel())

probabilities = LR.predict_proba(X_test)
print(probabilities[:15,:])


# Save the model using pickle
pickl = {'model': LR}
pickle.dump( pickl, open( 'Finalized_model123' + ".p", "wb" ) )

file_name = "finalized_model123.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

# Predicting the result
result=model.predict(X_test)
print(result)

# Checking the accuracy score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,result,normalize=True)
print(score)


