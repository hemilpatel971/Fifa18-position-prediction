# -*- coding: utf-8 -*-
"""
@author: HEMIL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

#reading the csv file
df = pd.read_csv(r'C:\Users\HEMIL\Desktop\New folder\CompleteDataset.csv')
df.head()

#taking all the fields which we need from the dataset
df = df[['Aggression','Crossing', 'Curve', 'Dribbling', 'Finishing','Free kick accuracy', 'Heading accuracy', 'Long shots',
                  'Penalties', 'Shot power', 'Volleys', 'Short passing', 'Long passing','Interceptions', 'Marking', 'Sliding tackle',
                  'Standing tackle','Strength', 'Vision', 'Acceleration', 'Agility', 'Reactions', 'Stamina', 'Balance', 'Ball control',
                  'Composure','Jumping','Sprint speed', 'Positioning','Preferred Positions']]

#removing GK position data as it is obvious
df = df[df['Preferred Positions'].str.strip() != 'GK']
df.isnull().values.any()
  
#find all the different positions on the field   
pos = df['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()
print(pos)


#there are multiple values for preferred positions columns and we need to split them and add them to the rows

# copy the structure
df_new = df.copy()
df_new.drop(df_new.index, inplace=True)

for i in pos:
    df_temp = df[df['Preferred Positions'].str.contains(i)]
    df_temp['Preferred Positions'] = i
    df_new = df_new.append(df_temp, ignore_index=True)
    
    
#adding some cell values which are in form of '72+5' instead of numerical values
cols = [col for col in df.columns if col not in ['Preferred Positions']]
for i in cols:
    df_new[i] = df_new[i].apply(lambda x: eval(x) if isinstance(x,str) else x)


#plotting the graph only for striker position
sns.set_style("darkgrid")
fig, ax = plt.subplots()
df_new_ST = df_new[df_new['Preferred Positions'] == 'ST'].iloc[::200,:-1]
df_new_ST.T.plot.line(color = 'black', figsize = (15,10), legend = False, ylim = (0, 110), title = "ST's attributes distribution", ax=ax)

ax.set_xlabel('Attributes')
ax.set_ylabel('Rating')

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(labels = cols, rotation=90)

for ln in ax.lines:
    ln.set_linewidth(1)

ax.axvline(0, color='red', linestyle='--')   
ax.axvline(12.9, color='red', linestyle='--')

ax.axvline(13, color='blue', linestyle='--')
ax.axvline(17, color='blue', linestyle='--')

ax.axvline(17.1, color='green', linestyle='--')
ax.axvline(28, color='green', linestyle='--')

ax.text(5, 100, 'Attack Attributes', color = 'red', weight = 'bold')
ax.text(13.5, 100, 'Defend Attributes', color = 'blue', weight = 'bold')
ax.text(22, 100, 'Mixed Attributes', color = 'green', weight = 'bold')


#FIRST WE JUST CLASSIFY THE PLAYERS IN TWO CATEGORYS ATTACK AND DEFENCE
#changing the values of preferred positions into numerical form
mapping = {'ST': 1, 'RW': 1, 'LW': 1, 'RM': 1, 'CM': 1, 'LM': 1, 'CAM': 1, 'CF': 1, 'CDM': 0, 'CB': 0, 'LB': 0, 'RB': 0, 'RWB': 0, 'LWB': 0}
df_new_bin = df_new.replace({'Preferred Positions': mapping})
X = df_new_bin.iloc[:,0:29].values
Y = df_new_bin.iloc[:,29].values

#splitting dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting logistic regression to the training set
clf = LogisticRegression(random_state = 0).fit(X_train,Y_train) 

#predicting the test set results
y_pred = clf.predict(X_test)
#y_new_pred = clf.predict([[82,	44,	56,	69,	39,	61,	88,	65,	70,	74,	58,	79,	85,	90,	84,	88,	88,	85,	74,	62,	60,	86,	73,	52,	75,	84,	85,	72,	38]])
Lr_acc = clf.score(X_test, Y_test)
print ('Logistic Regression Accuracy: {}'.format(Lr_acc))

#plotting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)

"""
#Logistic regression preformed after Doing PCA but the accuracy is relatively same as before

#Train test split
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X, Y, test_size = 0.2, random_state=0)

#feature scaling
sc = StandardScaler()
X_train_2 = sc.fit_transform(X_train_2)
X_test_2 = sc.transform(X_test_2)
#applying PCA
pca = PCA(n_components = 3)
X_train_2 = pca.fit_transform(X_train_2)
X_test_2 = pca.transform(X_test_2)
var = pca.explained_variance_ratio_

#fitting logistic regression to the training set after applying PCA

clf = LogisticRegression(random_state = 0).fit(X_train_2,Y_train_2) 

#predicting the test set results
y_pred_2 = clf.predict(X_test_2)
Lr_acc_pca = clf.score(X_test_2, Y_test_2)
print ('Logistic Regression Accuracy: {}'.format(Lr_acc_pca))
"""

#CLASSIFYING PLAYERS FOR ALL THE 14 POSITIONS


mapping_all = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CF': 7, 'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}
df_new_all = df_new.replace({'Preferred Positions': mapping_all})
#here only the dependent attribute changes which is Y from (1,0) to (0,13) and not the independent attributes which are X
Y_all = df_new_all.iloc[:,29].values


#splitting dataset into training and test set
X_train, X_test, Y_train_all, Y_test_all = train_test_split(X, Y_all, test_size = 0.2, random_state=0)


#fitting logistic regression to the training set
clf = LogisticRegression(random_state = 0).fit(X_train,Y_train_all) 


#predicting the test set results
y_pred_all = clf.predict(X_test)
#y_new_pred = clf.predict([[82,	44,	56,	69,	39,	61,	88,	65,	70,	74,	58,	79,	85,	90,	84,	88,	88,	85,	74,	62,	60,	86,	73,	52,	75,	84,	85,	72,	38]])
Lr_acc_all = clf.score(X_test, Y_test_all)
print ('Logistic Regression Accuracy: {}'.format(Lr_acc_all))

#plotting the confusion matrix
from sklearn.metrics import confusion_matrix
cm_all = confusion_matrix(Y_test_all,y_pred)
print(cm_all)

"""
The Accuracy to classify players drops significantly when we try classify from 2 categories to 14. 
Some attributes like crossing which is not Important for a striker but Important for a Winger or a Fullback.
players main foot data may also increase accuracy
""" 

















