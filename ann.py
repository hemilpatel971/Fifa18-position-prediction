# -*- coding: utf-8 -*-
"""
@author: HEMIL
"""

import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model


#reading the csv file
df = pd.read_csv(r'E:\Mygit\CompleteDataset.csv')
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
#print(pos)


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
    
    
mapping_all = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CF': 7, 'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}

df_new = df_new.replace({'Preferred Positions': mapping_all})
X = df_new.iloc[:,0:29].values
Y = df_new.iloc[:,29].values
Y_all = np_utils.to_categorical(Y)


#used later to convert the class number to actual position
class_names = list(mapping_all.keys())


#splitting dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_all, test_size = 0.2, random_state=0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#artificial neural network
classifier = Sequential()

#input layer and first hidden layer
classifier.add(Dense(output_dim = 21 ,init = 'uniform', activation = 'relu',input_dim = 29))

#second hidden layer
classifier.add(Dense(output_dim = 21 ,init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 14, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 50)
#accuracy when training it on 50 epochs or 100 remain roughly the same.

#accuracy is around 50% which is quite low and the reason is mentioned in the logistic_regression.py file


classifier.summary()

#saving weights
filename = (r"E:\Mygit\fifa18_ann.h5")
classifier.save(filename)

# Loading weights
fname = (r"E:\Mygit\fifa18_ann.h5")
model = load_model(filename)
print("model loaded")
#now to predict from the model which is loaded use model.predict()



# Predicting the Test set results

#y_pred gives the probability for all the classes a given set of attribute mat present 
y_pred = classifier.predict(X_test)
for i in y_pred:
    print(list(map('{:.6f}'.format,i)))


#round_pred rounds up the probability and gives the class number to which these particular attributes belong.
round_pred = classifier.predict_classes(X_test)
#round_pred = model.predict_classes(X_test)


#predicting the position from a new set of attributes
lwb = [44, 51,	46,	48,	34,	45,	41,	39,	36,	44,	33,	40,	38,	41,	49,	43,	48,	48,	40,	60,	47,	50,	52,	54,	28,	43,	65,	58,	44]
x = np.array(lwb)
#reshapeing the array
x = np.reshape(x, (1, 29))

"""
the reason why we need to make an array and then reshape it to a particular dimension is stated below.
https://stackoverflow.com/questions/55660786/valueerror-error-when-checking-input-expected-dense-1-input-to-have-shape-24
"""

#predicting position from new attributes
round_pred_test = classifier.predict_classes(x)
#round_pred_test = model.predict_classes(x)
m = max(round_pred_test)
print(class_names[m])



   

    


