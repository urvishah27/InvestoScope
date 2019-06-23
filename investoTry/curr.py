import numpy as np
import pandas as pd
import datetime
from dateutil import parser
from sklearn.model_selection import train_test_split
import io
from sklearn.svm import SVR
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly as py
import datetime
from dateutil import parser
from sklearn.metrics import r2_score

def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

def find_future_dates(n=10):
    future_dates = list()
    d1 = parser.parse('2019-03-05')
    holidays = [parser.parse('2019-03-21'),parser.parse('2019-04-17'),parser.parse('2019-04-19'),parser.parse('2019-05-01')]
    while True:
        if is_business_day(d1) and d1 not in holidays:
            future_dates.append(d1)
            n=n-1
            if n<=0:
                break
        d1 = d1 + datetime.timedelta(days=1)
    return np.array(future_dates)

def forex_get_model(curr1):
    curr2='INR'
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={curr1}&to_symbol=INR&apikey=PRSVB5B7LKOVVNAV&outputsize=full&datatype=csv'
    df = pd.read_csv(io.StringIO(requests.get(url).content.decode('utf8')))
    tempdf = df
    fig2 = {
        'data': [{'x':tempdf['timestamp'], 'y':tempdf['open'], 'name':'Open'},
                 {'x':tempdf['timestamp'], 'y':tempdf['close'], 'name':'Close'},
                 {'x':tempdf['timestamp'], 'y':tempdf['low'], 'name':'Low'},
                 {'x':tempdf['timestamp'], 'y':tempdf['high'], 'name':'High'},],
        'layout': {'autosize':True}
    }
    history_graph = py.offline.plot(fig2, include_plotlyjs=False, output_type='div', show_link=False)
    df.drop(['open','high','low'],axis=1,inplace=True)
    df.columns = ['Date','Close']
    df.index = df.Date
    df.drop('Date',1, inplace=True)
    dataset = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x, y = [], []
    for i in range(60,len(dataset)):
        x.append(scaled_data[i-60:i,0])
        y.append(scaled_data[i,0])
    x_test = np.array(x)[:100]
    y_test = np.array(y)[:100]
    x_train = np.array(x)[100:]
    y_train = np.array(y)[100:]
    y_train = y_train.reshape(y_train.shape[0],1)
    svmodel = SVR(kernel='linear',epsilon=0)
    svmodel.fit(x_train,y_train)
    svpreds = np.array(dataset.min() + ( (np.array(svmodel.predict(x_test))) * (dataset.max() - dataset.min() )))
    svactual = np.array(dataset.min() + ( (np.array(y_test)) * (dataset.max() - dataset.min() )))
    svpdo = pd.DataFrame({'Preds':svpreds,'Actual':svactual.reshape(svactual.shape[0],)}, index=df.index[:100])
    r2 = r2_score(svpdo['Actual'], svpdo['Preds'])
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(y.shape[0],1)
    svmodel = SVR(kernel='linear',epsilon=0)
    svmodel.fit(x,y)
    initial_x = np.array(y[:60])
    fig1 = {
        'data': [{'x':svpdo.index, 'y':svpdo['Preds'], 'name':'Predictions'},
                {'x':svpdo.index, 'y':svpdo['Actual'], 'name':'Actual'},],
        'layout': {'autosize':True}
    }
    compare_graph = py.offline.plot(fig1, include_plotlyjs=False, output_type='div', show_link=False)
    return svmodel,r2, history_graph, initial_x, compare_graph, dataset.min(), dataset.max()

models = {}
def forex_predict_graph(curr1='USD', pred_days=5):
    current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = None, None, None, None, None, None, None
    if models.get(curr1) is None:
        current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = forex_get_model(curr1)
        models[curr1] = {'model':current_model,'r2':r2, 'history_graph':history_graph, 'initial_x':initial_x, 'compare_graph':compare_graph, 'dataset_min':dataset_min, 'datset_max':dataset_max}
    else:
        current_model = models[curr1]['model']
        r2 = models[curr1]['r2']
        history_graph = models[curr1]['history_graph']
        initial_x = models[curr1]['initial_x']
        compare_graph = models[curr1]['compare_graph']
        dataset_min = models[curr1]['dataset_min']
        dataset_max = models[curr1]['dataset_max']
    next_n_dates = find_future_dates(pred_days)
    vals = list()
    for i in range(pred_days):
        new_y = current_model.predict(initial_x.reshape(1,60))
        vals.append(dataset_min+ ((new_y)*(dataset_max-dataset_min)))
        initial_x = initial_x[1:]
        initial_x = np.append(initial_x,new_y)
    vals = np.array(vals)
    preds_df = pd.DataFrame({'Predictions':vals.reshape(pred_days,),'Date':next_n_dates})
    fig1 = {
        'data': [{'x':preds_df['Date'], 'y':preds_df['Predictions'], 'name':'Predicted Closing Value'},],
        'layout': {'autosize':True}
    }
    preds_graph =  py.offline.plot(fig1, include_plotlyjs=False, output_type='div', show_link=False)
    return preds_graph, history_graph, compare_graph,r2

curr_to_try = ['USD','EUR','JPY']
for a in curr_to_try:
    x,y,z,j = forex_predict_graph(curr1=a)
    print(j)
