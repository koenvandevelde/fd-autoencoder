from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import datetime

#Load data
prefix = 'data/preprocessed/'
X_train_df = pd.read_pickle(prefix+'X_train.pkl')
X_validation_df = pd.read_pickle(prefix+'X_validation.pkl')
Y_validation_df = pd.read_pickle(prefix+'Y_validation.pkl')
X_test_df = pd.read_pickle(prefix+'X_test.pkl')
Y_test_df = pd.read_pickle(prefix+'Y_test.pkl')
X_train = X_train_df.values
X_validation = X_validation_df.values
Y_validation = Y_validation_df.values
X_test = X_test_df.values
Y_test = Y_test_df

#Variables
results_path_prefix = 'results/'
time = str(datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S'))
model_name = time + '_clustering_multivariate_gaussian'

#Train
print('TRAIN')
gmm = GaussianMixture(n_components=3, n_init=4, random_state=42)
gmm.fit(X_train)

#Validation
print('VALIDATION')
tresholds = np.linspace(-400, 0, 100)
Y_scores = gmm.score_samples(X_validation)
scores = []
for treshold in tresholds:
    y_hat = (Y_scores < treshold).astype(int)
    scores.append([recall_score(y_pred=y_hat, y_true=Y_validation),
                 precision_score(y_pred=y_hat, y_true=Y_validation),
                 fbeta_score(y_pred=y_hat, y_true=Y_validation, beta=2)])

scores = np.array(scores)
# print(scores[:, 2].max(), scores[:, 2].argmax())

#Test
print('TEST')
final_tresh = tresholds[scores[:, 2].argmax()]
y_hat_test = (gmm.score_samples(X_test) < final_tresh).astype(int)
cnf_matrix = confusion_matrix(Y_test, y_hat_test)


#Plot confusion matrix
LABELS = ["Fraud", "Normal"]
plt.figure('Clustering' + '_confusion matrix')
sns.heatmap(cnf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix ")
plt.ylabel('Predicted class')
plt.xlabel('True class')
plt.savefig(results_path_prefix + model_name)
plt.close()






print("---DONE---")
import time
for x in range(5):  
    time.sleep(3)
    print('\007')