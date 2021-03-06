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
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score
import sys

LABELS = ["Fraud", "Normal"]
validation_path_prefix = 'data/validation/'
results_path_prefix = 'results/'
models_path_prefix = 'models/'
models_path_postfix = '.h5'
model_name = '2019-07-13--22:08:24-ae-single-loss:mean_squared_error-optimizer:adam-encoding_dim:14-epoch:10-batch-size:32-regularizers-l1:0dot01_scaled_all:0_scaled_amount:1-model'
file_path = models_path_prefix + model_name + models_path_postfix
title = model_name

#####- Load trained model -#####
autoencoder = load_model(file_path, compile=False)
#

#Load datasets
datasets_path_prefix = 'data/preprocessed/'
scaled_amount = 1
if scaled_amount:
    datasets_path_prefix = 'data/preprocessed/scaled_amount'

scaled_all = 0
if scaled_all:
    datasets_path_prefix = 'data/preprocessed/scaled_all'

X_train_df = pd.read_pickle(datasets_path_prefix+'X_train.pkl')
Y_train = pd.read_pickle(datasets_path_prefix+'Y_train.pkl')
X_test_df = pd.read_pickle(datasets_path_prefix+'X_test.pkl')
Y_test = pd.read_pickle(datasets_path_prefix+'Y_test.pkl')
X_validation_df = pd.read_pickle(datasets_path_prefix+'X_validation.pkl')
Y_validation_df = pd.read_pickle(datasets_path_prefix+'Y_validation.pkl')



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
Y_validation = Y_validation_df.values
X_train = X_train_df.values
Y_train = Y_train.values
X_test = X_test_df.values


def describeDataFrame(df, header):
    # print(header)
    # print('head')
    # print(df.head(5))
    # print('describe')
    # print(df.describe())
    # print('tail')
    # print(df.tail(5))
    df.to_csv(validation_path_prefix + header + ".csv")

describeDataFrame(X_train_df.describe(), model_name + '_dataframe_training')

#Predictions on combined transactions of validation set
predictions = autoencoder.predict(X_validation)
reconstruction_error = X_validation - predictions
# print('error on first transaction')
# print(X_validation[0] - predictions[0])
columns = list(X_validation_df.columns) 
predictions_df = pd.DataFrame(reconstruction_error, columns=columns)
describeDataFrame(X_validation_df.describe(), model_name + "_dataframe_validation_original")
describeDataFrame(predictions_df.describe(), model_name + "_dataframe_validation_predictions")

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
describeDataFrame(predictions_df_fraud.describe(), model_name + "_predictions_dataframe_validation_type_fraud")
describeDataFrame(X_validation_fraud_df.describe(), model_name + "_ori_dataframe_validation_type_fraud")

#Predictions on non fraud transactions of validation set (low mse expected)
predictions = autoencoder.predict(X_validation_non_fraud)
reconstruction_error_non_fraud = X_validation_non_fraud - predictions
predictions_df_non_fraud = pd.DataFrame(reconstruction_error_non_fraud, columns=columns)
describeDataFrame(predictions_df.describe(), model_name + "_predictions_dataframe_validation_type_non-fraud")
describeDataFrame(predictions_df_non_fraud.describe(), model_name + "_ori_dataframe_validation_type_non-fraud")

#Predictions on fraud transactions of validation set (high mse expected)
reconstructionTitle = '__Reconstruction_fraud'
plt.figure(model_name + reconstructionTitle)
#Y = sum up error of all features. X = Range from zero to n fraud instances
plt.plot(range(len(reconstruction_error_fraud)), np.sum(reconstruction_error_fraud, axis = 1), lw=2, alpha=0.3)
plt.xlabel('Reconstruction error')
# plt.ylabel('FP')
plt.draw()
plt.savefig(results_path_prefix + model_name + reconstructionTitle)

#Predictions on non fraud transactions of validation set (high mse expected)
reconstructionTitle = '__Reconstruction_non_fraud'
plt.figure(model_name + reconstructionTitle)
#Y = sum up error of all features. X = Range from zero to n fraud instances
plt.plot(range(len(reconstruction_error_non_fraud)), np.sum(reconstruction_error_non_fraud, axis = 1), lw=2, alpha=0.3)
plt.xlabel('Reconstruction error')
# plt.ylabel('FP')
plt.draw()
plt.savefig(results_path_prefix + model_name + reconstructionTitle)



predictions = autoencoder.predict(X_validation)
reconstruction_error_fraud = X_validation - predictions
mse = np.mean(np.power(reconstruction_error_fraud, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'validation_set_Y': Y_validation})

#####- F1 scire as a single metric for PR curve -#####
# f1_score = f1_score(error_df.validation_set_Y, error_df.reconstruction_error, average='weighted')

#####- PR curve -#####
fscores = []
precision, recall, thresholds = precision_recall_curve(error_df.validation_set_Y, error_df.reconstruction_error)

print(precision)
# print('length')
# print(len(precision))
# print(len(recall))

for i in range(len(precision)):
    pr = precision[i]
    rec = recall[i]
    fscore = 2*(pr*rec)/(pr + rec)
    fscores.append(fscore)


pr_df = pd.DataFrame({'precision' : precision[1:],
                    'recall': recall[1:],
                    'f1score' : fscores[1:],
                    'threshold' : thresholds})

print('max scores')
max_f1_scores = pr_df.nlargest(50, 'f1score')
describeDataFrame(max_f1_scores.head(50), model_name +  '_max_f1_scores')
print(max_f1_scores)

area = auc(recall, precision)

plt.figure(title + '__pr')
plt.plot(recall, precision, lw=2, alpha=0.3)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR curve. Area under curve:' + str(area) + '_f1_score:' + str(f1_score))
plt.legend(loc='best')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.draw()
plt.savefig(results_path_prefix + model_name + '__pr')
plt.close()


plt.figure(title + '__pr_threshold')
axes = plt.axes()
plt.plot(thresholds, precision[1:], label="Precision",lw=2)
plt.plot(thresholds, recall[1:], label="Recall",lw=2)
axes.set_xlim([0,20])
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.savefig(results_path_prefix + model_name + '__pr_threshold')
plt.close()

#####- ROC curve -#####
fpr_keras, tpr_keras, thresholds_keras = roc_curve(error_df.validation_set_Y, error_df.reconstruction_error, drop_intermediate=False)
area = auc(fpr_keras, tpr_keras)

plt.figure(title + '__roc')
plt.plot([0, 1], [0, 1], linestyle='--', label='50/50 accuracy line')
plt.plot(fpr_keras, tpr_keras, lw=2, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (0, area))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve . Area under curve:' + str(area))
plt.legend(loc='best')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.draw()
plt.savefig(results_path_prefix + model_name + '__roc')
plt.close()

#####- Plot function -#####
def plot(x, y, xlabel, ylabel, title):
    plt.figure(title + 'title')
    plt.plot(x, y, lw=2, alpha=0.3, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.draw()
    title = title.replace('.', 'dot')
    print(results_path_prefix + model_name + title)
    plt.savefig(results_path_prefix + model_name + title)

#####- Confusion Matrix on Validation set -#####
def confusionMatrixValidation(threshold):
    pred_y = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.validation_set_Y, pred_y)
    # print('conf matrix')
    # print(conf_matrix)
    # print('TN')
    # print(conf_matrix[0][0])
    # print('FP')
    # print(conf_matrix[1][0])
    # print('TP')
    # print(conf_matrix[1][1])
    # print('FN')
    # print(conf_matrix[0][1])

    anomaly_conf_matrix = [conf_matrix[1][1], conf_matrix[0][1]],[conf_matrix[1][0], conf_matrix[0][0]]

    plt.figure(title + '_threshold_' + str(threshold) + '_confusion matrix')
    sns.heatmap(anomaly_conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix " + str(threshold))
    plt.ylabel('Predicted class')
    plt.xlabel('True class')

    # plt.draw()
    plt.savefig(results_path_prefix + model_name +'__confusion matrix_' + str(threshold))
    plt.close()

confusionMatrixValidation(15)
confusionMatrixValidation(14)
confusionMatrixValidation(13)
confusionMatrixValidation(12)
confusionMatrixValidation(11)
confusionMatrixValidation(10)
confusionMatrixValidation(9)
confusionMatrixValidation(8)
confusionMatrixValidation(7)
confusionMatrixValidation(6)
confusionMatrixValidation(5)
confusionMatrixValidation(5)
confusionMatrixValidation(4)
confusionMatrixValidation(3)
confusionMatrixValidation(2)
confusionMatrixValidation(1)


TP = []
TN = []
FP = []
FN = []
precision = []
recall = []
tresholds = []

for x in np.arange(0, 10, 1):
    threshold_fixed = x
    tresholds.append(threshold_fixed)
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.validation_set_Y, pred_y)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(error_df.validation_set_Y, pred_y, pos_label=1, drop_intermediate=False)
    plot(fpr_keras, tpr_keras, 'fpr', 'tpr', '_ROC'  + str(x))
    # print('fpr_keras')
    # print(fpr_keras)
    # print('tpr_keras')
    # print(tpr_keras)
    # print('debugging pred_y')
    # print(pred_y)
    precision, recall, thresholds = precision_recall_curve(error_df.validation_set_Y, pred_y)
    print('debugging precision')
    print(precision)
    plot(recall, precision, 'recall', 'precision', '_PR'  + str(round(x)))
    tp = conf_matrix[1][1]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]
    FP.append(fp)
    TP.append(tp)
    TN.append(conf_matrix[0][0])
    FN.append(fn)
    # Precision = tp / (tp + fp)
    # print('precision tp:' + str(tp) + 'fp: ' + str(fp) + 'FN: ' + str(fn) +  ' threshold fixed: '  + str(threshold_fixed))
    # print(Precision)
    # precision.append(Precision)
    # Recall = tp / (tp + fn)
    # print(Recall)
    # recall.append(Recall)


mse_tresholdtwo = tresholds.index(2)
mse_tresholdfour = tresholds.index(4)
mse_tresholdsix = tresholds.index(6)
mse_tresholdten = tresholds.index(9)
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


#####- Amount analysis -#####
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


#####- Classification values for certrain thresholds -#####

# print('missed transactions', missed)
# print('treshold')
# for i in range(len(tresholds)): 
#     print (str(i) + ' - ' + str(tresholds[i]))
# print('TN')
# for i in range(len(TP)): 
#     print (str(i) + ' - ' + str(TP[i]))
# print('FP')   
# for i in range(len(FP)): 
#     print (str(i) + ' - ' + str(FP[i]))

#####- Thresholds FP tradeoff -#####
falsePositiveTitle = '__False_Positives'
plt.figure(model_name + falsePositiveTitle)
plt.plot(tresholds, FP, lw=2, label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('FP')
plt.draw()
plt.savefig(results_path_prefix + model_name + falsePositiveTitle)

#####- Thresholds FN tradeoff -#####
falseNegativeTitle = '__False_Negatives'
plt.figure(model_name + falseNegativeTitle)
plt.plot(tresholds, FN, lw=2, label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('FN')
plt.draw()
plt.savefig(results_path_prefix + model_name + falseNegativeTitle)

#####- Thresholds TP tradeoff -#####
truePositiveTitle = '__True_Positives'
plt.figure(model_name + truePositiveTitle)
plt.plot(tresholds, TP, lw=2, label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('TP')
plt.draw()
plt.savefig(results_path_prefix + model_name + truePositiveTitle)

#####- Thresholds TN tradeoff -#####
trueNegativeTitle = '__True_Negatives'
plt.figure(model_name + trueNegativeTitle)
plt.plot(tresholds, TN, lw=2, label='temp')
plt.xlabel('MSE Threshold')
plt.ylabel('TN')
plt.draw()
plt.savefig(results_path_prefix + model_name + trueNegativeTitle)

#####- Combine classification metrics and threshold tradeoff -#####
tresholdsTitle = '__tresholds'
plt.figure(model_name + tresholdsTitle)
plt.plot(tresholds, TN, lw=2,
             label='TN')
plt.plot(tresholds, TP, lw=2,
             label='TP')
plt.plot(tresholds, FP, lw=2,
             label='FP')
plt.plot(tresholds, FN, lw=2,
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
plt.plot(precision, lw=2)
plt.plot(recall, lw=2)
plt.xlabel('precision')
plt.ylabel('recall')
plt.draw()
plt.savefig(results_path_prefix + model_name + PRtitle)
plt.close()


tresholdsTitlePositives = '__positives'
plt.figure(model_name + tresholdsTitlePositives)
plt.plot(tresholds, TP, lw=2,label='TP')
plt.plot(tresholds, FP, lw=2,label='FP')
plt.legend([ 'TP', 'FP'], loc='upper left')                       
plt.xlabel('MSE Threshold')
plt.ylabel('TN')
plt.draw()
plt.savefig(results_path_prefix + model_name + tresholdsTitlePositives)


#####- End Validation Data. Start Test Data -#####
describeDataFrame(X_test_df.describe(), model_name + '_dataframe_test')
predictions = autoencoder.predict(X_test)
reconstruction_error_fraud = X_test - predictions
mse = np.mean(np.power(reconstruction_error_fraud, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'test_set_Y': Y_test})

#####- Confusion Matrix -#####
def confusionMatrixTest(threshold):
    pred_y = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.test_set_Y, pred_y)
    # print('conf matrix')
    # print(conf_matrix)
    # print('TN')
    # print(conf_matrix[0][0])
    # print('FP')
    # print(conf_matrix[1][0])
    # print('TP')
    # print(conf_matrix[1][1])
    # print('FN')
    # print(conf_matrix[0][1])

    anomaly_conf_matrix = [conf_matrix[1][1], conf_matrix[0][1]],[conf_matrix[1][0], conf_matrix[0][0]]

    plt.figure(title + '_threshold_' + str(threshold) + '_confusion matrix')
    sns.heatmap(anomaly_conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix " + str(threshold))
    plt.ylabel('Predicted class')
    plt.xlabel('True class')

    # plt.draw()
    plt.savefig(results_path_prefix + model_name + '_confusion matrix_test_' + str(threshold))
    plt.close()

confusionMatrixTest(4)
confusionMatrixTest(5)
confusionMatrixTest(6)
confusionMatrixTest(7)
confusionMatrixTest(8)
confusionMatrixTest(9)
confusionMatrixTest(10)
confusionMatrixTest(11)
confusionMatrixTest(12)
confusionMatrixTest(13)

#show all plots
#plt.show()
# sys.exit()

print("---DONE---")
import time
for x in range(5):  
    time.sleep(3)
    print('\007')