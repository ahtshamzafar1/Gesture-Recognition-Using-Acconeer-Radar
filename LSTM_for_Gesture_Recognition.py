# !/usr/bin/python3
#######################################
# Author: GESTAR Project Team
# Reusable with attribution: GESTAR, 2022
#######################################
"""
This is a Long Short Term Memory Network model for gesture recognition and classification. 
"""

import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv1D, MaxPool1D, Dropout,LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.metrics import categorical_crossentropy
import warnings
import numpy as np

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.regularizers import l2


import matplotlib.pyplot as plt
################ PARAMETERS ########################
path = 'data620x80/'
dataset= []
classes= []
label= []
classes= os.listdir(path)
sweeps=80
samples=620

############## DARA

button_press = os.listdir(path + 'button_press/')
for i, gestures in enumerate(button_press):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'button_press/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         # gmax=np.max(gesture)
         # gesture =gesture/gmax 
         dataset.append(np.array(gesture))
         label.append(0)
         
         

hand_away = os.listdir(path + 'hand_away/')
for i, gestures in enumerate(hand_away):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'hand_away/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         # gmax=np.max(gesture)
         # gesture =gesture/gmax 
         dataset.append(np.array(gesture))
         label.append(1)
         

hand_close = os.listdir(path + 'hand_close/')
for i, gestures in enumerate(hand_close):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'hand_close/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         # gmax=np.max(gesture)
         # gesture =gesture/gmax
         dataset.append(np.array(gesture))
         label.append(2)


         
         
no_gesture = os.listdir(path + 'no_gesture/')
for i, gestures in enumerate(no_gesture):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'no_gesture/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         # gmax=np.max(gesture)
         # gesture =gesture/gmax
         dataset.append(np.array(gesture))
         label.append(3)    

swipe_left = os.listdir(path + 'swipe/')
for i, gestures in enumerate(swipe_left):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (gestures.split('.')[1] == 'npy'):
         gesture= np.load(path + 'swipe/' + gestures)
         gesture = gesture.reshape((sweeps,samples))
         # gmax=np.max(gesture)
         # gesture =gesture/gmax
         dataset.append(np.array(gesture))
         label.append(4)         
         
###################### npy array ################         
dataset = np.array(dataset)
classes = np.array(classes)

print(dataset.shape)
print(classes.shape)

############## split into train and test ###############

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, Y_test = train_test_split(dataset,label, test_size = 0.30, random_state = 42)

############## togetagorical ######

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, len(classes))
y_test = to_categorical(Y_test, len(classes))

############## CNN Model ########
INPUT_SHAPE = (sweeps,samples)   # (207,50) samples, sweeps 

################ 77###################

model = Sequential()



# 2 LSTM layers
model.add(LSTM(64, input_shape=INPUT_SHAPE, return_sequences=True))
model.add(LSTM(64))

# dense layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(len(classes), activation='softmax'))



#Do not use softmax for binary classification
#Softmax is useful for mutually exclusive classes, either cat or dog but not both.
#Also, softmax outputs all add to 1. So good for multi class problems where each
#class is given a probability and all add to 1. Highest one wins. 

#Sigmoid outputs probability. Can be used for non-mutually exclusive problems.
#But, also good for binary mutually exclusive (cat or not cat). 

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy']) #adam,adadelta,SGD(lr=0.01),Adam(learning_rate=0.001),sparse_categorical_crossentropy,categorical_crossentropy, mean_squared_error
print(model.summary()) 



############# Plot model ########
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model.png')

############################### Training #################

# simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
mc = ModelCheckpoint('weights620x80lstm1.h5', verbose=True, save_best_only=True)

history = model.fit(X_train,  
                         y_train, 
                         batch_size =32, 
                         verbose = 1, 
                         epochs =5000, 
                         #validation_split=0.30,
                         validation_data=(X_test,y_test),
                         shuffle = True,
                         callbacks=[reduce_lr, early_stop,mc]

                     )


#model.save('fgestures.h5') 

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


###################### Model evalute ##############
score = model.evaluate(X_train, y_train)
print("Test loss:", score[0]) 
print("Test accuracy:", score[1])

########### classes prob ######
proba = model.predict(X_test)  #Get probabilities for each class
sorted_categories = np.argsort(proba[0])[:-11:-1]  #Get class names for top 10 categories                                
    
################ Accuracy score #####
prediction = model.predict(X_test) 
pred_x=np.argmax(prediction, axis=1)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, pred_x)

print('Accuracy Score = ', accuracy) 




from sklearn.metrics import confusion_matrix
import seaborn as sns
#Print confusion matrix
cm = confusion_matrix(Y_test, pred_x)
fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12,12))
plt.bar(np.arange(len(classes)), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(len(classes)), classes) 
