# Loan Predictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("loan_training.csv")
# preprocess the data
df["Loan_Status"] = df.Loan_Status.map({'Y':1, 'N':0}).astype(int)
#Gender Encoding
df = df.replace({"Gender":{"Male":1, "Female":0}})
df = df.replace({"Married":{"Yes":1, "No":0}})
# df["Dependents"] = df['Dependents'].replace('3+', '3')
# df["Dependents"] = pd.to_numeric(df['Dependents'], errors='coerce')
a = df['Self_Employed'].value_counts()
df = df.replace({"Self_Employed":{"Yes":1, "No":0}})
df['Education'].value_counts()
df=df.replace({"Education":{"Graduate":1, "Not Graduate":0}})
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})

df.fillna(df.median,inplace=True)
columns = df.columns
for column in columns:
  df[column] = pd.to_numeric(df[column], errors='coerce')

# Correlation to see if there's relationship between our variables. 
sns.heatmap(df.corr(),annot=True)

def correlationdrop(df, sl):
  columns = df.columns
  for column in columns:
      C=abs(df[column].corr(df['Loan_Status']))
      if C < sl:
        df=df.drop(columns=[column])
  return df

df= correlationdrop(df,0.05)
df.drop("Loan_ID", axis=1, inplace=True)
# print(df)
# Visualize the data
df.Property_Area[df.Loan_Status==1].value_counts(normalize = True).plot(kind='bar', alpha = 0.5)
plt.title('Loan Accepted by Property Area')
# plt.show()


df = df.dropna(subset=['Married'])
df = df.dropna(subset=['Credit_History'])
null_rows = df.loc[df['Credit_History'].isnull()]

print(df.columns)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
sc = MinMaxScaler()
X= sc.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

df.dropna(how="any")



classifier = SVC(kernel = 'rbf', gamma= 0.2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)




#make a prediction


# Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))