import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def shapeData(data, transform_amount, transform_all):
    if transform_amount:
        print('transforming amount')
        # scaler = StandardScaler()
        # print('scale data')
        # std_scale = scaler.fit(data['Amount'].values)
        # data = std_scale.transform(data)
        #data['Amount'] = std_scale.transform(data['Amount'].values.reshape(-1,1))
        #TODO refactor
        print('debug')
        print((data['Amount'].values.reshape(-1,1)))
        data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

        
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
    Y_validation.to_pickle(prefix+ 'Y_validation.pkl')
    X_validation.to_pickle(prefix+ 'X_validation_complete.pkl')
    X_validation = X_validation.drop(['Class'], axis=1)
    X_validation.to_pickle(prefix+ 'X_validation.pkl')
    Y_test = X_test['Class']
    Y_test.to_pickle(prefix+ 'Y_test.pkl')
    X_test = X_test.drop(['Class'], axis=1)
    X_train.to_pickle(prefix + 'X_train.pkl')
    Y_train.to_pickle(prefix + 'Y_train.pkl')
    X_test.to_pickle(prefix + 'X_test.pkl')


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

# data = pd.read_csv("data/creditcard.csv")
# shapeData(data, 1, 0)
# plotColumnValues('after_scaling',data, 'Amount')