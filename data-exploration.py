import pandas as pd
import matplotlib.pyplot as plt


df_all = pd.read_csv("data/creditcard.csv")
df_non_fraudulent = df_all[df_all.Class == 0]

# print('describe dataset')
# print(df.head(5))
# print(df.describe())
# print(df.tail(5))


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

def plotColumnHistogram(dfType, df, column_title):
    column = column_title
    title = column + '_' + dfType + '_non-fraudulent_exploratory_histogram'
    fig = plt.figure(title)
    plt.hist(pd.to_numeric(df[column]))
    plt.title(title)
    plt.ylabel(column)
    plt.draw()
    plt.savefig('data/exploration/' + title)
    plt.close()


def analyse(dfType, df):
    for x in range(1, 29):
        print('describe')
        dataFrame = df['V' + str(x)]
        print(dataFrame.describe())
        column = 'V' + str(x)
        plotColumnValues(dfType, df, column)
        plotColumnHistogram(dfType, df, column)
    plotColumnValues(dfType, df, 'Time')
    plotColumnHistogram(dfType, df, 'Time')
    plotColumnValues(dfType, df, 'Amount')
    plotColumnHistogram(dfType, df, 'Amount')
    plotColumnValues(dfType, df, 'Class')
    plotColumnHistogram(dfType, df, 'Class')

analyse('all', df_all)
analyse('non-fraudulent', df_non_fraudulent)

