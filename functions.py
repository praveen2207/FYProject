from datetime import date
from statsmodels.tsa.arima_model import ARIMA
from pandas import Series
import numpy as np
import pandas
import pickle
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from datetime import timedelta

# def load_dataset():
url = "prsaf1.csv"
names = ['pm2.5', 'dew_point', 'temperature', 'pressure', 'windspeed', 'result']
dataset = pandas.read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:, 0:5]
Y = array[:, 5]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'


def auto_reg(d1, d2):
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return np.array(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # load dataset
    series = Series.from_csv('prsaf3.csv', header=None)
    # seasonal difference
    X = series.values
    days_in_year = 365
    differenced = difference(X, days_in_year)
    differenced = differenced.astype(float)  # astype converts np.ndarray to the given type
    # multi-step out-of-sample forecast
    model_fit = pickle.load(open('trained_arima.sav','rb'))
    start_index = len(differenced)
    end_index = start_index + d2
    forecast = model_fit.predict(start=start_index, end=end_index)
    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    days = [[], []]
    d = date.today()
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        if day > d1:
            days[0].append(str(d.day) + "-" + str(d.month) + "-" + str(d.year))
            days[1].append(inverted)
            d += timedelta(days=1)
        print('Day %d: %f' % (day, inverted))
        history.append(inverted)
        day += 1
    plt.plot(history, color='blue')
    plt.savefig("auto-predic.png")
    return [days, inverted]


def display():
    # load_dataset()
    # shape
    dis1 = dataset.shape
    # column names
    dis2 = dataset.columns
    # head
    dis3 = str(dataset.head(2))
    # tail
    dis4 = dataset.tail(2)
    # descriptions
    dis5 = dataset.describe()
    # box and whisker plots
    dataset.plot(kind='box', subplots=True)
    plt.savefig("box.png")
    # histograms
    dataset.hist(figsize=(8, 8))
    plt.savefig("hist.png")
    # scatter plot matrix
    scatter_matrix(dataset[['pm2.5', 'dew_point', 'temperature', 'pressure', 'windspeed']], figsize=(8, 8))
    plt.savefig("scat.png")
    return [dis1, dis2, dis3, dis4, dis5]

def algo_comp():
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("algo.png")

def log_reg(pm, dew_point, temperature, pressure, windspeed):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(LogisticRegression(), X_train, Y_train, cv=kfold, scoring=scoring)
    example_measures = np.array([pm, dew_point, temperature, pressure, windspeed])
    example_measures = example_measures.reshape(1, -1)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    predictions = lr.predict(X_validation)
    prediction = float(lr.predict(example_measures))
    return prediction
