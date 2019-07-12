import datetime
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense #prefix this with tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import sys
   
LABELS = ["Normal", "Fraud"]

#Initialize variables in data and testset
#moved to data-preprocessing.py
# def initData():
#     df = pd.read_csv("data/creditcard.csv")
#     print(df.describe())
#     data = df.drop(['Time'], axis=1)
#     training_length = int(len(data)*0.9)
#     validation_length = int(len(data)*0.05)

#     X_train = data.iloc[:training_length]
#     #only use non fraudulent transactions for training set
#     X_train = X_train[X_train.Class == 0]
#     #drop class label for training set
#     X_train = X_train.drop(['Class'], axis=1) 
#     print("Number of frauds removed from training set: ")
#     print(training_length - len(X_train)) 

#     X_validation = data.iloc[training_length:training_length+validation_length]
#     X_test = data.iloc[training_length+validation_length:]


#     #list of all class labels only
#     Y_validation = X_validation['Class']
#     Y_validation.to_pickle('Y_validation.pkl')
#     X_validation = X_validation.drop(['Class'], axis=1)
#     X_validation.to_pickle('X_validation.pkl')
#     Y_test = X_test['Class']
#     Y_test.to_pickle('Y_test.pkl')
#     X_test = X_test.drop(['Class'], axis=1)
#     X_train.to_pickle('X_train.pkl')
#     X_test.to_pickle('X_test.pkl')

#initData()

prefix = 'data/preprocessed/'
X_train = pd.read_pickle(prefix+'X_train.pkl')

# X_test = pd.read_pickle(prefix+'X_test.pkl')
# Y_test = pd.read_pickle(prefix+'Y_test.pkl')
X_validation = pd.read_pickle(prefix+'X_validation.pkl')
Y_validation = pd.read_pickle(prefix+'Y_validation.pkl')
print(X_train.describe())
X_train = X_train.values
# X_test = X_test.values
X_validation = X_validation.values

#####- Building model -#####
input_dim = X_train.shape[1]


nb_epoch = 35
batch_size = 32
encoding_dim = 8
type='default'
optimizer = 'adam'
loss = 'mean_squared_error'
regularizerInput= 10e-5
activity_regularizer=regularizers.l1(regularizerInput)
title = str(datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')) + '-ae-' + type + '-loss:' + loss + '-optimizer:' + optimizer + '-encoding_dim:' + str(encoding_dim) + '-epoch:' + str(nb_epoch) + '-batch-size:'+ str(batch_size) + '-regularizers-l1:' + str(regularizerInput)
title = title.replace('.', 'dot')

if type == 'default':
    from autoencoder_models.default import run
elif type == 'sixlayer':
    from autoencoder_models.sixlayer import run
elif type == 'fourlayer':
    from autoencoder_models.fourlayer import run
elif type == 'single':
    from autoencoder_models.single import run
else:
    print('unknown algorithm')
autoencoder = run(input_dim, encoding_dim, activity_regularizer)


#Once your model looks good, configure its learning process with .compile()
autoencoder.compile(optimizer=optimizer, 
                    loss=loss, 
                    metrics=['accuracy'])

filePath="models/"+title+"-model.h5"
checkpointer = ModelCheckpoint(filepath=filePath,
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

#You can now iterate on your training data in batches:
#contains:
#loss
#val accuracy
#val loss
#accuracy
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_validation, X_validation),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

#A Keras model instance. 
#If an optimizer was found as part of the saved model, the model is already compiled. 
#Otherwise, the model is uncompiled and a warning will be displayed. 
#When compile is set to False, the compilation is omitted without any warning.
#see https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
# autoencoder = load_model(filePath)
# pathPrefix = 'results/'

# #Model evaluation
# #####- Loss -#####
fig = plt.figure(title + '__loss')
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.draw()
plt.show()
# plt.savefig(pathPrefix + title + '__loss')


# #####- Accuracy -#####
# fig = plt.figure(title + '__accuracy')
# plt.plot(history['accuracy'])
# plt.plot(history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.draw()
# plt.savefig(pathPrefix + title + '__accuracy')

# scores = autoencoder.evaluate(X_train, X_train)
# #print('ok')
# #print("\n%s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))
# #print('/ok')

# #make predictions on new data/test data
# predictions = autoencoder.predict(X_test)
# mse = np.mean(np.power(X_test - predictions, 2), axis=1)
# error_df = pd.DataFrame({'reconstruction_error': mse,
#                         'validation_set_X': X_validation,
#                         'validation_set_Y': Y_validation})
# #print('error_df description')
# #print(error_df.describe())

# #####- ROC -#####
# print('real values')
# print(X_test[0])
# print(X_test[1])
# print('before predictions')
# print(predictions[0])
# print(predictions[1])
# #take all the rows but keep the second column
# #predictions = autoencoder.predict(X_test)[:, 1]
# print('after predictions')
# print(predictions[0])
# print('y test 0')
# #print(Y_test[0])
# print('predictions 0')
# print(predictions[0])
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(error_df.Y_validation, error_df.reconstruction_error)
# auc_keras = auc(fpr_keras, tpr_keras)

# plt.figure(pathPrefix + title + '__roc')
# plt.plot([0, 1], [0, 1], linestyle='--', label='50/50 accuracy line')
# plt.plot(fpr_keras, tpr_keras, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.2f)' % (0, auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.draw()
# plt.savefig(pathPrefix + title + '__roc')

# #####- Confusion Matrix -#####
# ##################################################TODO choose threshold
# threshold_fixed = 5
# pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
# conf_matrix = confusion_matrix(error_df.true_class, pred_y)
# print(conf_matrix)

# plt.figure(title + '__confusion matrix')
# sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
# plt.title("Confusion matrix")
# plt.ylabel('True class')
# plt.xlabel('Predicted class')
# plt.draw()
# plt.savefig(pathPrefix + title + '__confusion matrix')


# #show all plots
# #plt.show()



print("---DONE---")
import time
for x in range(5):  
    time.sleep(3)
    print('\007')