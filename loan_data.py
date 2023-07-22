import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# function that does preprocessing for the Data
def preprocess_data():
    df = pd.read_csv("loan_training.csv")
    df["Loan_Status"] = df.Loan_Status.map({'Y':1, 'N':0}).astype(int)
    df = df.replace({"Gender":{"Male":1, "Female":0}})
    df = df.replace({"Married":{"Yes":1, "No":0}})
    df["Dependents"] = df['Dependents'].replace('3+', '3')
    df["Dependents"] = pd.to_numeric(df['Dependents'], errors='coerce')
    a = df['Self_Employed'].value_counts()
    df = df.replace({"Self_Employed":{"Yes":1, "No":0}})
    df['Education'].value_counts()
    df=df.replace({"Education":{"Graduate":1, "Not Graduate":0}})
    df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})

    df.fillna(df.median,inplace=True)
    columns = df.columns
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
    df = df.dropna(subset=['Married'])
    df = df.dropna(subset=['Credit_History'])
    df = df.dropna(subset=['Gender'])
    df = df.dropna(axis=1)
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    sc = MinMaxScaler()
    X= sc.fit_transform(x)
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)
    classifier = SVC(kernel = 'rbf', gamma= 0.2)
    classifier.fit(X_train, y_train)
    return classifier
    
def predict_res(clsf, single_sample):
    y = clsf.predict(single_sample)
    return y
    