from statsmodels.tsa.arima_model import ARIMA
from pandas import Series
import numpy as np
import pickle


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


series = Series.from_csv('prsaf3.csv', header=None)
X = series.values;
days_in_year = 365;
differenced = difference(X, days_in_year)
differenced = differenced.astype(float)
model = ARIMA(differenced, order=(7, 0, 1))
model_fit = model.fit(disp=0)
filename = 'trained_arima.sav'
pickle.dump(model_fit, open(filename, 'wb'))