import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn import metrics

from data_reader_rewritten import load_data

#The following is an example MLP network used to classify all-cause mortality given data from the MIMIC-III dataset
#reshaping is only necessary because of the MLP - most models won't require the data.reshape() lines

#Load in the raw data
data_dir = '.\\data\\'
train_raw = load_data(data_dir, mode='train')
val_raw = load_data(data_dir, mode='validation')

#Format training and validation data for network use
train_data = np.asarray(train_raw[0])
train_data = train_data.reshape((-1, 3648))
train_labels = np.asarray(train_raw[1])

val_data = np.asarray(val_raw[0])
val_data = val_data.reshape((-1, 3648))
val_labels = np.asarray(val_raw[1])

#Build the machine learning model
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC','accuracy'])

#Train the model. history contains the AUC and accuracy over epochs, which can be plotted to find how the
#model improves over training
history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=36, verbose=1)

#Plot how the model improves over training (on the validation set)
plt.plot(history.history['accuracy'])
plt.title("Training accuracy over epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

print()
print("Testing on test set...")

#Load and process the test set
test_raw = load_data(data_dir, mode='test')
test_data = np.asarray(test_raw[0])
test_data = test_data.reshape((-1, 3648))
test_labels = np.asarray(test_raw[1])

#Get the predictions of the trained model on the test set
predicted_labels = model.predict(test_data)

#Find the final ROC AUC for the model
auc = metrics.roc_auc_score(test_labels, predicted_labels)

print("AUC ROC on test set = "+str(auc))

plt.show()
