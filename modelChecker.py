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

title = str(datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S'))
title = title.replace('.', 'dot')

LABELS = ["Normal", "Fraud"]

autoencoder = load_model('vae.h5', compile=False)
pathPrefix = 'results/'


X_train = pd.read_pickle('x_train.pkl')
X_test = pd.read_pickle('x_test.pkl')
y_test = pd.read_pickle('y_test.pkl')

X_train = X_train.values
X_test = X_test.values

#Model evaluation
#####- Loss -#####
    

#scores = autoencoder.evaluate(X_train, X_train)
#print('ok')
#print("\n%s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))
#print('/ok')

#make predictions on new data/test data
predictions = autoencoder.predict(X_test)
temp = X_test - predictions
print('verschil')
print(temp)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
print('error_df description')
print(error_df.describe())

#####- ROC -#####
print('real values')
print(X_test[0])
print(X_test[1])
print('before predictions')
print(predictions[0])
print(predictions[1])
#take all the rows but keep the second column
#predictions = autoencoder.predict(X_test)[:, 1]
print('after predictions')
print(predictions[0])
print('y test 0')
print(y_test[0])
print('predictions 0')
print(predictions[0])
fpr_keras, tpr_keras, thresholds_keras = roc_curve(error_df.true_class, error_df.reconstruction_error)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(pathPrefix + title + '__roc')
plt.plot([0, 1], [0, 1], linestyle='--', label='50/50 accuracy line')
plt.plot(fpr_keras, tpr_keras, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (0, auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.draw()
plt.savefig(pathPrefix + title + '__roc')

#####- Confusion Matrix -#####
##################################################TODO choose threshold
threshold_fixed = 5
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, pred_y)
print(conf_matrix)

plt.figure(title + '__confusion matrix')
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.draw()
plt.savefig(pathPrefix + title + '__confusion matrix')


#show all plots
#plt.show()

import time
for x in range(15):  
    time.sleep(3)
    print('\007')