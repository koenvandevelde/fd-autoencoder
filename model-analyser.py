from __future__ import division  #python 2 doesn't automatically convert ints to float on division
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
validation_path_prefix = 'data/validation/'
results_path_prefix = 'results/'
models_path_prefix = 'models/'
models_path_postfix = '.h5'
model_name = '2019-07-07--15:42:24-ae-single-loss:mean_squared_error-optimizer:adam-encoding_dim:8-epoch:35-batch-size:32-regularizers-l1:0dot0001-model'
file_path = models_path_prefix + model_name + models_path_postfix
title = model_name

#Load trained model
autoencoder = load_model(file_path, compile=False)
#

#Load datasets
datasets_path_prefix = 'data/preprocessed/'
X_train = pd.read_pickle(datasets_path_prefix+'X_train.pkl')
X_test = pd.read_pickle(datasets_path_prefix+'X_test.pkl')
Y_test = pd.read_pickle(datasets_path_prefix+'Y_test.pkl')
X_validation_df = pd.read_pickle(datasets_path_prefix+'X_validation.pkl')
Y_validation = pd.read_pickle(datasets_path_prefix+'Y_validation.pkl')



X_validation_complete = pd.read_pickle(datasets_path_prefix+'X_validation_complete.pkl')

X_validation_fraud_df = X_validation_complete[X_validation_complete.Class == 1]
Y_validation_fraud = X_validation_fraud_df['Class'].values
X_validation_fraud_df = X_validation_fraud_df.drop(['Class'], axis=1)
X_validation_fraud = X_validation_fraud_df.values

X_validation_non_fraud_df = X_validation_complete[X_validation_complete.Class == 0]
Y_validation_non_fraud = X_validation_non_fraud_df['Class'].values
X_validation_non_fraud_df = X_validation_non_fraud_df.drop(['Class'], axis=1)
X_validation_non_fraud = X_validation_non_fraud_df.values


X_validation = X_validation_df.values
X_train = X_train.values
X_test = X_test.values


def describeDataFrame(df, header):
    # print(header)
    # print('head')
    # print(df.head(5))
    # print('describe')
    # print(df.describe())
    # print('tail')
    # print(df.tail(5))
    df.describe().to_csv(validation_path_prefix + header + ".csv")



#Predictions on combined transactions of validation set
predictions = autoencoder.predict(X_validation)
reconstruction_error = X_validation - predictions
# print('error on first transaction')
# print(X_validation[0] - predictions[0])
columns = list(X_validation_df.columns) 
predictions_df = pd.DataFrame(reconstruction_error, columns=columns)
describeDataFrame(X_validation_df, model_name + "_dataframe_validation_original")
describeDataFrame(predictions_df, model_name + "_dataframe_validation_predictions")

#Predictions on fraud transactions of validation set (high mse expected)
predictions = autoencoder.predict(X_validation_fraud)
reconstruction_error_fraud = X_validation_fraud - predictions
sum = np.sum(reconstruction_error_fraud, axis = 1)
# temp = np.matrix([[1, 2], [3, 4]])
# lol = np.sum(temp, axis = 1)
# print(lol)
# print('should be 7 ')
# print('end test')
predictions_df_fraud = pd.DataFrame(reconstruction_error_fraud, columns=columns)
describeDataFrame(predictions_df_fraud, model_name + "_predictions_dataframe_validation_type_fraud")
describeDataFrame(X_validation_fraud_df, model_name + "_ori_dataframe_validation_type_fraud")

#Predictions on non fraud transactions of validation set (low mse expected)
predictions = autoencoder.predict(X_validation_non_fraud)
reconstruction_error_non_fraud = X_validation_non_fraud - predictions
predictions_df_non_fraud = pd.DataFrame(reconstruction_error_non_fraud, columns=columns)
describeDataFrame(predictions_df, model_name + "_predictions_dataframe_validation_type_non-fraud")
describeDataFrame(predictions_df_non_fraud, model_name + "_ori_dataframe_validation_type_non-fraud")

#Predictions on fraud transactions of validation set (high mse expected)
reconstructionTitle = '__Reconstruction_fraud'
plt.figure(model_name + reconstructionTitle)
#Y = sum up error of all features. X = Range from zero to n fraud instances
plt.plot(range(len(reconstruction_error_fraud)), np.sum(reconstruction_error_fraud, axis = 1), lw=1, alpha=0.3)
plt.xlabel('Reconstruction error')
# plt.ylabel('FP')
plt.draw()
plt.savefig(results_path_prefix + model_name + reconstructionTitle)

#Predictions on non fraud transactions of validation set (high mse expected)
reconstructionTitle = '__Reconstruction_non_fraud'
plt.figure(model_name + reconstructionTitle)
#Y = sum up error of all features. X = Range from zero to n fraud instances
plt.plot(range(len(reconstruction_error_non_fraud)), np.sum(reconstruction_error_non_fraud, axis = 1), lw=1, alpha=0.3)
plt.xlabel('Reconstruction error')
# plt.ylabel('FP')
plt.draw()
plt.savefig(results_path_prefix + model_name + reconstructionTitle)

# colums = [1,2,3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]


# column = 'mse'
# title = column + ' histogram'
# fig = plt.figure(title, figsize=(20, 10))

# ar = np.array(colums)
# width = 0.35  # the width of the bars
# columns = ar - 2*width
# # plt.bar(columns, error[0], width, color='blue', align='center')
# columns = ar - width
# # plt.bar(colums, error[1], width, color='red', align='center')
# columns = ar + width
# # plt.bar(colums, error[2], width, color='purple', align='center')
# columns = ar + 2*width
# # plt.bar(colums, error[3], width, color='green', align='center')
# plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
# plt.title(column)
# plt.ylabel(column)
# plt.draw()
# plt.legend()
# df = pd.DataFrame({'instance 1': error[0], 'instance 2': error[2], 'instance 3': error[3], 'instance 4': error[4]}, index=colums)
# ax = df.plot.bar(rot=0)
# plt.savefig('data/exploration/' + title)

# sys.exit()
mse = np.mean(np.power(reconstruction_error, 2), axis=1)
print('mse')
print(mse)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'validation_set_Y': Y_validation})

print(error_df.reconstruction_error.describe())

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
#print(y_test[0])
print('predictions 0')
print(predictions[0])
fpr_keras, tpr_keras, thresholds_keras = roc_curve(error_df.validation_set_Y, error_df.reconstruction_error)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(title + '__roc')
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
plt.savefig(results_path_prefix + model_name + '__roc')

#####- Confusion Matrix -#####
def confusionMatrix(threshold):
    pred_y = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.validation_set_Y, pred_y)
    print('conf matrix')
    print(conf_matrix)
    print('TN')
    print(conf_matrix[0][0])
    print('FP')
    print(conf_matrix[1][0])
    print('TP')
    print(conf_matrix[1][1])
    print('FN')
    print(conf_matrix[0][1])

    plt.figure(title + '__confusion matrix')
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    # plt.draw()
    plt.savefig(results_path_prefix + model_name + str(threshold)+'__confusion matrix')
    plt.close()

confusionMatrix(5)
confusionMatrix(4)
confusionMatrix(3)
confusionMatrix(2)
confusionMatrix(1)


TP = []
TN = []
FP = []
FN = []
precision = []
recall = []
tresholds = []

for x in np.arange(0, 10, 0.1):
    threshold_fixed = x
    tresholds.append(threshold_fixed)
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.validation_set_Y, pred_y)
    tp = conf_matrix[1][1]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]
    FP.append(fp)
    TP.append(tp)
    TN.append(conf_matrix[0][0])
    FN.append(fn)
    Precision = tp / (tp + fp)
    print('precision tp:' + str(tp) + 'fp: ' + str(fp) + 'FN: ' + str(fn) +  ' threshold fixed: '  + str(threshold_fixed))
    print(Precision)
    precision.append(Precision)
    Recall = tp / (tp + fn)
    print(Recall)
    recall.append(Recall)




# precision = TP / (TP + FP)
# recall = TP / (TP + FN)


mse_tresholdtwo = tresholds.index(2)
mse_tresholdfour = tresholds.index(4)
mse_tresholdsix = tresholds.index(6)
mse_tresholdten = tresholds.index(9.9)
print('tresholds')
print(FP[mse_tresholdtwo])
print(TP[mse_tresholdtwo])
print(FN[mse_tresholdtwo])
print(TN[mse_tresholdtwo])

print(FP[mse_tresholdfour])
print(TP[mse_tresholdfour])
print(FN[mse_tresholdfour])
print(TN[mse_tresholdfour])

print(FP[mse_tresholdsix])
print(TP[mse_tresholdsix])
print(FN[mse_tresholdsix])
print(TN[mse_tresholdsix])

print(FP[mse_tresholdten])
print(TP[mse_tresholdten])
print(FN[mse_tresholdten])
print(TN[mse_tresholdten])
print('debugging')
print(pred_y[0])
print(error_df.validation_set_Y.values[0])
missed = 0
correct = 0
for i in range(len(pred_y)):
    if pred_y[i] == 1:
        if error_df.validation_set_Y.values[i] == 0:
            missed = missed + 1
            print('missed transaction with amount')
            print(X_validation[i][28])
        else:
            correct = correct + 1
            print('Correctly classified transaction with amount')
            print(X_validation[i][28])


print('missed transactions', missed)
print('treshold')
for i in range(len(tresholds)): 
    print (str(i) + ' - ' + str(tresholds[i]))
print('TN')
for i in range(len(TP)): 
    print (str(i) + ' - ' + str(TP[i]))
print('FP')   
for i in range(len(FP)): 
    print (str(i) + ' - ' + str(FP[i]))

falsePositiveTitle = '__False_Positives'
plt.figure(model_name + falsePositiveTitle)
plt.plot(tresholds, FP, lw=1, alpha=0.3,
             label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('FP')
plt.draw()
plt.savefig(results_path_prefix + model_name + falsePositiveTitle)

falseNegativeTitle = '__False_Negatives'
plt.figure(model_name + falseNegativeTitle)
plt.plot(tresholds, FN, lw=1, alpha=0.3,
             label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('FN')
plt.draw()
plt.savefig(results_path_prefix + model_name + falseNegativeTitle)

truePositiveTitle = '__True_Positives'
plt.figure(model_name + truePositiveTitle)
plt.plot(tresholds, TP, lw=1, alpha=0.3,
             label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('TP')
plt.draw()
plt.savefig(results_path_prefix + model_name + truePositiveTitle)

trueNegativeTitle = '__True_Negatives'
plt.figure(model_name + trueNegativeTitle)
plt.plot(tresholds, TN, lw=1, alpha=0.3,
             label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('TN')
plt.draw()
plt.savefig(results_path_prefix + model_name + trueNegativeTitle)

tresholdsTitle = '__tresholds'
plt.figure(model_name + tresholdsTitle)
plt.plot(tresholds, TN, lw=1, alpha=0.3,
             label='TN')
plt.plot(tresholds, TP, lw=1, alpha=0.3,
             label='TP')
plt.plot(tresholds, FP, lw=1, alpha=0.3,
             label='FP')
plt.plot(tresholds, FN, lw=1, alpha=0.3,
             label='FN')
plt.legend(['TN', 'TP', 'FP', 'FN'], loc='upper left')                       
plt.xlabel('MSE Threshold')
plt.ylabel('Classification metric')
plt.draw()
plt.savefig(results_path_prefix + model_name + tresholdsTitle)


# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
PRtitle = '__PR'
plt.figure(model_name + PRtitle)
plt.plot(precision, lw=1, alpha=0.3)
plt.plot(recall, lw=1, alpha=0.3)
plt.xlabel('precision')
plt.ylabel('recall')
plt.draw()
plt.savefig(results_path_prefix + model_name + PRtitle)
plt.close()


tresholdsTitlePositives = '__positives'
plt.figure(model_name + tresholdsTitlePositives)
plt.plot(tresholds, TP, lw=1, alpha=0.3,
             label='TP')
plt.plot(tresholds, FP, lw=1, alpha=0.3,
             label='FP')
plt.legend([ 'TP', 'FP'], loc='upper left')                       
plt.xlabel('MSE Threshold')
plt.ylabel('TN')
plt.draw()
plt.savefig(results_path_prefix + model_name + tresholdsTitlePositives)

#show all plots
#plt.show()

print("---DONE---")
import time
for x in range(5):  
    time.sleep(3)
    print('\007')