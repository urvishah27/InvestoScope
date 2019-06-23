from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from .mls import get_model, crypto_get_model, forex_get_model, find_future_dates
from .models import Purchase
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
models = {}

def stock_predict_graph(stock='IOC.BSE', pred_days=30):
    current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = None, None, None, None, None, None, None
    if models.get(stock) is None:
        current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = get_model(stock)
        models[stock] = {'model':current_model,'r2':r2, 'history_graph':history_graph, 'initial_x':initial_x, 'compare_graph':compare_graph, 'dataset_min':dataset_min, 'dataset_max':dataset_max}
    else:
        current_model = models[stock]['model']
        r2 = models[stock]['r2']
        history_graph = models[stock]['history_graph']
        initial_x = models[stock]['initial_x']
        compare_graph = models[stock]['compare_graph']
        dataset_min = models[stock]['dataset_min']
        dataset_max = models[stock]['dataset_max']
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
    return preds_graph, history_graph, compare_graph, r2

def forex_predict_graph(curr1='USD', pred_days=30):
    current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = None, None, None, None, None, None, None
    if models.get(curr1) is None:
        current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = forex_get_model(curr1)
        models[curr1] = {'model':current_model,'r2':r2, 'history_graph':history_graph, 'initial_x':initial_x, 'compare_graph':compare_graph, 'dataset_min':dataset_min, 'dataset_max':dataset_max}
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

def crypto_predict_graph(curr1='BTC', pred_days=30):
    current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = None, None, None, None, None, None, None
    if models.get(curr1) is None:
        current_model,r2, history_graph, initial_x, compare_graph, dataset_min, dataset_max = crypto_get_model(curr1)
        models[curr1] = {'model':current_model,'r2':r2, 'history_graph':history_graph, 'initial_x':initial_x, 'compare_graph':compare_graph, 'dataset_min':dataset_min, 'dataset_max':dataset_max}
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


def index(request):
    return render(request, 'index.html')


@login_required
def customLogout(request):
    logout(request)
    return HttpResponseRedirect("/")

def newsView(request):
    return render(request, 'news.html')
def stockSearch(request, toSearchStock, stockName=''):
    try:
        preds_graph, history_graph, compare_graph, r2 = stock_predict_graph(stock=toSearchStock,pred_days = 30)
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={toSearchStock}&apikey=GL94EUS0R586G6IN'
        results = requests.get(url).json()["Global Quote"]
        stock_high = results['03. high']
        stock_low = results["04. low"]
        stock_change = results['09. change']
        stock_changeper = results['10. change percent']
        stock_open = results['02. open']
        stock_prevclose = results['08. previous close']
        stock_price = results['05. price']
        context = {
        'history_graph':history_graph,
        'preds_graph':preds_graph,
        'compare_graph':compare_graph,
        'r2':r2,
        'stock_code':toSearchStock,
        'stock_name':stockName,
        'stock_high':stock_high,
        'stock_low':stock_low,
        'stock_change':stock_change,
        'stock_changeper':stock_changeper,
        'stock_open':stock_open,
        'stock_prevclose':stock_prevclose,
        'stock_price':stock_price
        }
        return render(request, 'result.html', context)
    except:
        return render(request, 'result.html', context={})

def forexSearch(request, toSearchForex):
    try:
        preds_graph, history_graph, compare_graph, r2 = forex_predict_graph(curr1=toSearchForex,pred_days = 30)
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={toSearchForex}&to_currency=INR&apikey=5RX3ZR73LN5JNIA0'
        results = requests.get(url).json()["Realtime Currency Exchange Rate"]
        forex_rate = results["5. Exchange Rate"]
        forex_name = results['2. From_Currency Name']
        context = {
        'history_graph':history_graph,
        'preds_graph':preds_graph,
        'compare_graph':compare_graph,
        'r2':r2,
        'forex_rate':forex_rate,
        'forex_name':forex_name,
        'forex_code':toSearchForex,
        }
        return render(request, 'result_crypto.html', context)
    except:
        return render(request, '500.html')

def cryptoSearch(request, toSearchForex):
    try:
        preds_graph, history_graph, compare_graph, r2 = crypto_predict_graph(curr1=toSearchForex,pred_days = 30)
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={toSearchForex}&to_currency=INR&apikey=GL94EUS0R586G6IN'
        results = requests.get(url).json()["Realtime Currency Exchange Rate"]
        forex_rate = results["5. Exchange Rate"]
        forex_name = results['2. From_Currency Name']
        context = {'history_graph':history_graph,
        'pred_graph30':pred_graph30,
        'compare_graph':compare_graph,
        'r2':r2,
        'forex_rate':forex_rate,
        'forex_name':forex_name,
        'forex_code':toSearchForex,
        }
        return render(request, 'result_crypto.html', context)
    except:
        return render(request, '500.html')


def findDataAboutPurchases(all_purchases):
    symbols = list()
    purchase_ids = list()
    per_unit = list()
    quantity = list()
    investment_price = list()
    stock_type = list()
    stock_url = list()
    for i in all_purchases:
        symbols.append(i.stock)
        purchase_ids.append(i.id)
        per_unit.append(i.per_unit_price)
        quantity.append(i.units)
        investment_price.append(i.get_investment())
        stock_type.append(i.stock_type)
        stock_url.append(f'stock/{i.stock}')
    return zip(purchase_ids, symbols, per_unit, quantity, investment_price, stock_type, stock_url)

    

@login_required
def portfolio(request):
    logged_user = request.user
    print(request.user.email)
    all_purchases = Purchase.objects.filter(user=logged_user)
    allPurchaseData = findDataAboutPurchases(all_purchases)
    api_keys = ['Q1SINE3MUKKLZ790','0K4F7XX2QSAPMEBK','GE78DQVZHQH6NLJ0','E6SMZDKRU8QONCDA','PRSVB5B7LKOVVNAV', 'CCQ2DKTVMREEUGUC', 'PVCSGN3X8593BLDH','Q6HL4L3S1NI0247M','5RX3ZR73LN5JNIA0','P6YI5Z075H874SHD']
    cost_price = 0 
    usymbols = {}

    for purchase_ids, symbols, per_unit, quantity, investment_price, stock_type, stock_url in allPurchaseData:
        cost_price = cost_price + (per_unit*quantity)
        if usymbols.get(symbols) is None:
            usymbols[symbols] = [per_unit,stock_type]
        else:
            usymbols[symbols][0] = usymbols[symbols][0] + per_unit
    anum = 0
    sell_price = 0.0
    days_profit = 0.0
    for a in usymbols.keys():
        aunit = usymbols[a][0]
        print(aunit)
        atype = usymbols[a][1]
        akey = api_keys[anum]
        if atype=='Stock' or atype=='stock':
            url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={a}&apikey={akey}'
            result = requests.get(url).json()["Global Quote"]
            aprice = result['05. price']
            aopen = result['02. open']
            aclose = aprice
            sell_price = sell_price + (float(aprice) * float(aunit))
            days_profit = days_profit + (float(auint)*float(aclose-aopen))
        elif atype=='Forex' or atype=='forex' or atype=='Crypto' or atype=='crypto':
            url=f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={a}&to_currency=INR&apikey={akey}'
            result = requests.get(url).json()["Realtime Currency Exchange Rate"]
            aprice = result['5. Exchange Rate']
            sell_price = sell_price + ((aprice) * (aunit))
        anum = anum+1
        if anum >= 10:
            anum = 0
    context = {
        'allPurchaseData': allPurchaseData,
        'total_profit':(sell_price - cost_price),
        'days_profit':days_profit,
        'net_worth':sell_price
    }
    return render(request, 'portfolio.html', context)



@login_required
def addPurchase(request):
    if request.method == 'POST':
        stock = request.POST.get('form_stock')
        user = User.objects.get(email=request.POST.get('user_email'))  
        print(user.email)
        units = request.POST.get('form_units')
        per_unit_price = request.POST.get('form_per_unit_price')
        stock_type = request.POST.get('form_stock_type')
        new_purchase = Purchase.objects.create(
            stock = stock,
            user = user,
            units = units,
            per_unit_price = per_unit_price,
            stock_type = stock_type
        )
        new_purchase.save()
        return HttpResponseRedirect("/portfolio")
    else:
        return render(request, '500.html')


def signUpView(request):
    if request.method == 'POST':
        entered_email = request.POST.get('sign_email')
        passwd = request.POST.get('sign_password')
        usname = request.POST.get('sign_username')
        if User.objects.filter(email=entered_email).exists() or User.objects.filter(username=usname).exists():
            return HttpResponseRedirect("/login")
        else:
            new_use = User.objects.create(email=entered_email,username=usname,password=passwd)
            new_use.save()
            login(request, new_use)
            return HttpResponseRedirect("/portfolio")
    else:
        return render(request, '500.html')

def loginCheckView(request):
    if request.method == 'POST':
        passwd = request.POST.get('log_password')
        usname = request.POST.get('log_username')
        if User.objects.filter(username=usname).exists():
            user = authenticate(username=usname, password=passwd)
            if user is None:
                return HttpResponseRedirect("/login")
            else:
                login(request, user)
                return HttpResponseRedirect("/portfolio")
        else:
            return HttpResponseRedirect("/login")
    else:
        return render(request, '500.html')

def loginView(request):
    return render(request, 'auth-login.html')

def signUpView(request):
    return render(request, 'auth-register.html')


@login_required
def delPurchase(request, delId):
    if request.method == 'POST':
        a = request.POST.get('id_to_del')
        if a == delId:
            Purchase.objects.filter(id=a).delete()
            return HttpResponseRedirect("/portfolio")
        else:
            return render(request, '500.html')
    else:
        return render(request, '500.html')