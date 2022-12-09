from django.shortcuts import render
from . import choices

### For ML
import os 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
###

csv_filename = os.path.join(os.path.dirname(__file__), 'train.csv')
a = ["KMeans"]

def index(request):
    a[0] = "KMeans"
    context = {
        'PoolArea': choices.PoolArea, 
        'LotArea': choices.LotArea, 
        'TotalBsmtSF': choices.TotalBsmtSF,
        'OverallQual': choices.OverallQual,
        'Price_to_sell': choices.price_to_sell
    }
    return render(request, 'pages/index.html', context) 

def about(request):
    return render(request, 'pages/about.html')

def forest(request):
    a[0] = "RandomForest"
    context = {
        'MSZoning': choices.MSZoning,  
        'Utilities': choices.Utilities, 
        'LotConfig': choices.LotConfig,
        'HouseStyle': choices.HouseStyle,
        'GarageType': choices.GarageType
    }
    return render(request, 'pages/forest.html', context) 

def lassocv(request):
    a[0] = "LassoCV"
    context = {
        'MSZoning': choices.MSZoning,  
        'PoolArea': choices.PoolArea, 
        'LotArea': choices.LotArea, 
        'TotalBsmtSF': choices.TotalBsmtSF,
        'OverallQual': choices.OverallQual,
    }
    return render(request, 'pages/lassocv.html', context)    

def search(request):
    if a[0] == "KMeans":
        print("It's",a[0],"mode")
        train_df = pd.read_csv(csv_filename)
        features = ['LotArea', 'PoolArea', 'TotalBsmtSF', 'OverallQual', 'SalePrice']
        X = train_df[features]
        model = KMeans(n_clusters=8, random_state=42)
        model.fit(X)
        LotArea = 0
        PoolArea = 0 
        TotalBsmtSF = 0
        OverallQual = 0
        SalePrice = 0
        if 'price' in request.GET:
            SalePrice = request.GET['price']
        if 'quality' in request.GET:
            OverallQual = request.GET['quality']
        if 'pool' in request.GET:
            PoolArea = request.GET['pool']
        if 'LotArea' in request.GET:
            LotArea = request.GET['LotArea']
        if 'basement' in request.GET:
            TotalBsmtSF = request.GET['basement']
        tmp = {'LotArea': LotArea, 'PoolArea': PoolArea, 'TotalBsmtSF': TotalBsmtSF, 'OverallQual': OverallQual, 'SalePrice': SalePrice}
        response = pd.DataFrame(data=tmp, index=[0])
        preds = model.predict(response)
        print(preds[0])
        return render(request, 'pages/search.html')
    elif a[0] == "RandomForest":
        print("It's",a[0],"mode")
        MSZoning = 'A'
        Utilities = 'AllPub'
        LotConfig = 'Inside'
        HouseStyle = '1Story'
        GarageType = '2Types'
        if 'MSZoning' in request.GET:
            SalePrice = request.GET['MSZoning']
        if 'Utilities' in request.GET:
            OverallQual = request.GET['Utilities']
        if 'LotConfig' in request.GET:
            PoolArea = request.GET['LotConfig']
        if 'HouseStyle' in request.GET:
            LotArea = request.GET['HouseStyle']
        if 'GarageType' in request.GET:
            TotalBsmtSF = request.GET['GarageType']
        tmp = {'MSZoning': MSZoning, 'Utilities': Utilities, 'LotConfig': LotConfig, 'HouseStyle': HouseStyle, 'GarageType': GarageType}
        response = pd.DataFrame(data=tmp, index=[0])
        print(response)
        train_df = pd.read_csv(csv_filename)
        features = ['MSZoning', 'Utilities', 'LotConfig', 'HouseStyle', 'GarageType']
        X = pd.get_dummies(train_df[features])        
        y = train_df["MSZoning"]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X, y)

        X = pd.get_dummies(response)
        print(X)
        preds = model.predict(X)
        print(preds)
        return render(request, 'pages/search.html')
        
    elif a[0] == "LassoCV":
        print("It's",a[0],"mode")
        
    return render(request, 'pages/search.html')
