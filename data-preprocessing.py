import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def shapeData(data, transform_amount, transform_all, path='_'):
    if transform_amount:
        print('transforming amount')
        data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    if transform_all:
        print('transforming all')
        # scaler = StandardScaler()
        # print('scale data')
        # std_scale = scaler.fit(data['Amount'].values)
        # data = std_scale.transform(data)
        #data['Amount'] = std_scale.transform(data['Amount'].values.reshape(-1,1))
        #TODO refactor
        print('debug')
        print((data['Amount'].values.reshape(-1,1)))
        data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
        data['V1'] = StandardScaler().fit_transform(data['V1'].values.reshape(-1, 1))
        data['V2'] = StandardScaler().fit_transform(data['V2'].values.reshape(-1, 1))
        data['V3'] = StandardScaler().fit_transform(data['V3'].values.reshape(-1, 1))
        data['V4'] = StandardScaler().fit_transform(data['V4'].values.reshape(-1, 1))
        data['V5'] = StandardScaler().fit_transform(data['V5'].values.reshape(-1, 1))
        data['V6'] = StandardScaler().fit_transform(data['V6'].values.reshape(-1, 1))
        data['V7'] = StandardScaler().fit_transform(data['V7'].values.reshape(-1, 1))
        data['V8'] = StandardScaler().fit_transform(data['V8'].values.reshape(-1, 1))
        data['V9'] = StandardScaler().fit_transform(data['V9'].values.reshape(-1, 1))
        data['V10'] = StandardScaler().fit_transform(data['V10'].values.reshape(-1, 1))
        data['V11'] = StandardScaler().fit_transform(data['V11'].values.reshape(-1, 1))
        data['V12'] = StandardScaler().fit_transform(data['V12'].values.reshape(-1, 1))
        data['V13'] = StandardScaler().fit_transform(data['V13'].values.reshape(-1, 1))
        data['V14'] = StandardScaler().fit_transform(data['V14'].values.reshape(-1, 1))
        data['V15'] = StandardScaler().fit_transform(data['V15'].values.reshape(-1, 1))
        data['V16'] = StandardScaler().fit_transform(data['V16'].values.reshape(-1, 1))
        data['V17'] = StandardScaler().fit_transform(data['V17'].values.reshape(-1, 1))
        data['V18'] = StandardScaler().fit_transform(data['V18'].values.reshape(-1, 1))
        data['V19'] = StandardScaler().fit_transform(data['V19'].values.reshape(-1, 1))
        data['V20'] = StandardScaler().fit_transform(data['V20'].values.reshape(-1, 1))
        data['V21'] = StandardScaler().fit_transform(data['V21'].values.reshape(-1, 1))
        data['V22'] = StandardScaler().fit_transform(data['V22'].values.reshape(-1, 1))
        data['V23'] = StandardScaler().fit_transform(data['V23'].values.reshape(-1, 1))
        data['V24'] = StandardScaler().fit_transform(data['V24'].values.reshape(-1, 1))
        data['V25'] = StandardScaler().fit_transform(data['V25'].values.reshape(-1, 1))
        data['V26'] = StandardScaler().fit_transform(data['V26'].values.reshape(-1, 1))
        data['V27'] = StandardScaler().fit_transform(data['V27'].values.reshape(-1, 1))
        data['V28'] = StandardScaler().fit_transform(data['V28'].values.reshape(-1, 1))

        
    data = data.drop(['Time'], axis=1)
    training_length = int(len(data)*0.9)
    validation_length = int(len(data)*0.05)

    X_train = data.iloc[:training_length]

    Y_train = X_train['Class']
    #only use non fraudulent transactions for training set
    X_train = X_train[X_train.Class == 0]
    #drop class label for training set
    X_train = X_train.drop(['Class'], axis=1) 
    print("Number of frauds removed from training set: ")
    print(training_length - len(X_train)) 

    X_validation = data.iloc[training_length:training_length+validation_length]
    X_test = data.iloc[training_length+validation_length:]


    prefix = 'data/preprocessed/'

    
    #list of all class labels only
    Y_validation = X_validation['Class']
    Y_validation.to_pickle(prefix+ path +'Y_validation.pkl')
    X_validation.to_pickle(prefix+ path +  'X_validation_complete.pkl')
    X_validation = X_validation.drop(['Class'], axis=1)
    X_validation.to_pickle(prefix+ path+  'X_validation.pkl')
    Y_test = X_test['Class']
    Y_test.to_pickle(prefix+ path+ 'Y_test.pkl')
    X_test = X_test.drop(['Class'], axis=1)
    X_train.to_pickle(prefix + path+  'X_train.pkl')
    Y_train.to_pickle(prefix + path +'Y_train.pkl')
    X_test.to_pickle(prefix + path+ 'X_test.pkl')


def plotColumnValues(dfType, df, column_title):
    column = column_title
    title = column + '_' + dfType + '_non-fraudulent_exploratory_analysis'
    fig = plt.figure(title)
    plt.plot(pd.to_numeric(df[column]))
    plt.title(title)
    plt.ylabel(column)
    plt.draw()
    plt.savefig('data/exploration/' + title)
    plt.close()


data = pd.read_csv("data/creditcard.csv")
shapeData(data, 0, 0)
plotColumnValues('before_scaling',data, 'Amount')

data = pd.read_csv("data/creditcard.csv")
shapeData(data, 1, 0, 'scaled_amount')
plotColumnValues('after_scaling',data, 'Amount')

data = pd.read_csv("data/creditcard.csv")
shapeData(data, 0, 1, 'scaled_all')
plotColumnValues('after_scaling',data, 'Amount')