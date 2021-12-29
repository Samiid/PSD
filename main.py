import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Train.csv')
dftest = pd.read_csv('Test.csv')
df.head()

print('Mising Value pada setiap atribut:')
print(df.isna().sum())



df['Segmentation'].value_counts()
plt.subplots(figsize=(10,5))
sns.countplot(df['Segmentation'].sort_values())
plt.xlabel('Target Label')
plt.ylabel('Jumlah')
plt.title('Perbandingan Target Label')
plt.show()

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df['Var_1'] = labelencoder.fit_transform(df['Var_1'])
df['Segmentation'] = labelencoder.fit_transform(df['Segmentation'])
df['Gender'] = labelencoder.fit_transform(df['Gender'])
df['Ever_Married'] = labelencoder.fit_transform(df['Ever_Married'])
df['Spending_Score'] = labelencoder.fit_transform(df['Spending_Score'])
df['Graduated'] = labelencoder.fit_transform(df['Graduated'])
df = pd.get_dummies(df, columns = ['Profession'])
df = pd.get_dummies(df, columns = ['Var_1'])

df.dropna(axis=0, inplace=True)

df.drop(columns="ID", inplace=True)

Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)
atas = Q3 + 1.5 * IQR

mean = round(df["Age"].mean())
for i in df.index:
    if(df.loc[i]["Age"] > atas):
         df.loc[i, "Age"] = mean
df['Age'].plot.box()

Q1 = df['Work_Experience'].quantile(0.25)
Q3 = df['Work_Experience'].quantile(0.75)
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)
atas = Q3 + 1.5 * IQR

mean = round(df["Work_Experience"].mean())
for i in df.index:
    if(df.loc[i]["Work_Experience"] > atas):
         df.loc[i, "Work_Experience"] = mean
df['Work_Experience'].plot.box()

Q1 = df['Family_Size'].quantile(0.25)
Q3 = df['Family_Size'].quantile(0.75)
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)
atas = Q3 + 1.5 * IQR

mean = round(df["Family_Size"].mean())
for i in df.index:
    if(df.loc[i]["Family_Size"] > atas):
         df.loc[i, "Family_Size"] = mean
df['Family_Size'].plot.box()

features = df.drop(columns='Segmentation')
X = features
Y = df.Segmentation

from imblearn.over_sampling import SMOTE
smote = SMOTE('auto')
X_sm, y_sm = smote.fit_resample(X,Y)
print(X_sm.shape, y_sm.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=None, svd_solver='full')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
hasil_variance = pca.explained_variance_ratio_

from sklearn.model_selection import cross_val_score
scores = cross_val_score(neigh, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
neigh = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto', leaf_size=30, p=4, metric='euclidean', metric_params=None, n_jobs=None)
neigh.fit(X_train, y_train)
pred = neigh.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
predlr = lr.predict(X_test)
print(classification_report(y_test,predlr))
print(confusion_matrix(y_test,predlr))
