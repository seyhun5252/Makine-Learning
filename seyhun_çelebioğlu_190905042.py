# -*- coding: utf-8 -*-
"""Copy of Seyhun_Çelebioğlu_190905042.ipynb



Kullandığım Kütüphaneler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

"""Verileri exel dosyasında çekip yazdırma"""

df = pd.read_csv("Covid19-Turkey.csv")

df

"""Burada columnların ne olduğunu öğnrendim"""

df.columns

"""burada nan değerlerin yerine sıfır atama işlemi yaptım"""

# for column
df['DailyCases'] = df['DailyCases'].replace(np.nan, 0)

# for whole dataframe
df = df.replace(np.nan, 0)

# inplace
df.replace(np.nan, 0, inplace=True)

df.head()

"""İnfo komutu burada sütunlar hakkında bilgi aldım"""

df.info()

"""Describe methodu ile verilerin istatiksel olarak çıkardım"""

df.describe().T

df.head()

"""Burada verilerden kaç tane var işlemi yapıldı"""

df.DailyCases.value_counts()

df.TotalCases.value_counts()

df.TotalRecovered.value_counts()

df.TotalDeaths.value_counts()

"""Burada veriler görselleştirildi """

ds1=df["TotalCases"]
ds1.plot()

ds2=df["DailyCases"]
ds2.plot()

ds3=df["TotalDeaths"]
ds3.plot()

ds4=df["TotalRecovered"]
ds4.plot()

ds5=df["ActiveCases"]
ds5.plot()

ds6=df["Caseincrase rate %"]
ds6.plot()

ds3=df["TotalDeaths"]
ds1=df["TotalCases"]
ds1.plot()
ds3.plot()

ds3=df["TotalDeaths"]
ds1=df["TotalCases"]
ds4=df["TotalRecovered"]
ds1.plot()
ds3.plot()
ds4.plot()

ds1=df["TotalCases"]
ds2=df["DailyCases"]
ds3=df["TotalDeaths"]
ds4=df["TotalRecovered"]
ds1.plot()
ds2.plot()
ds3.plot()
ds4.plot()

plt.plot(df['DailyCases'])
plt.plot(df['TotalCases'])

plt.title("Covid-19 Graph in TURKEY", fontsize=15)
plt.xlabel("Number of days", fontsize=15)
plt.ylabel("Cases", fontsize=15)
plt.legend(["Daily Cases","Total Cases"])

plt.plot(df['TotalDeaths'],color ='red')
plt.plot(df['TotalRecovered'], color ="green")
plt.title("Covid-19 Graph in TURKEY", fontsize=15)
plt.xlabel("Number of days", fontsize=15)
plt.ylabel("Cases", fontsize=15)
plt.legend(["Total Deaths","Total Recovered"])

plt.pie([13531,106673,43373,35510],labels=["Mart","Nisan","Mayıs","Haziran"],autopct='%1.1f%%',shadow=True)
plt.title("Belirlenen Vaka Sayısının Aylara Göre Dağılımı")
plt.show()

plt.figure(figsize=(9,3))
plt.scatter(df["Date"],df["Caseincrase rate %"])
plt.plot(df["Date"],df["Caseincrase rate %"])
plt.xlabel("Geçen Gün")
plt.ylabel("%")
plt.title("Vaka Artış Yüzdesi")

"""burada hangi columlar kullanılacak diye seçildi"""

df.columns

df = df[["DailyCases","TotalCases","TotalDeaths","TotalRecovered","ActiveCases","DailyTest Cases","TotalIntensive Care","Intubated Cases",
         "Caseincrase rate %","Daily(Cases/Test) %","(Active Cases / Population) %"]]

df

"""Bağımlı ve Bağımsız değişkenleri ayırdık"""

x = df.drop(["DailyCases"], axis = 1)
y = df["DailyCases"]

"""Eğitim ve Test verisinin ayrılması"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 144)

""" GridSearchCV"""

params = {"colsample_bytree":[0.4,0.5,0.6],
         "learning_rate":[0.01,0.02,0.09],
         "max_depth":[2,3,4,5,6],
         "n_estimators":[100,200,500,2000]}

xgb = XGBRegressor()

grid = GridSearchCV(xgb, params, cv = 10, n_jobs = -1, verbose = 2)

grid.fit(x_train, y_train)

grid.best_params_

"""En uygun parametreleri giriyoruz"""

xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.01, max_depth = 4, n_estimators = 2000)

"""Modelimizi eğitiyoruz"""

model_xgb = xgb1.fit(x_train, y_train)

"""Tahmin yapıyoruz"""

model_xgb.predict(x_test)[15:20]

"""Tahmin edilen  ve gerçek verileri karşılaştırabiliriz"""

y_test[15:20]

"""modelin skorunu hesapladık. 0-1 arası değer döner bize"""

model_xgb.score(x_test, y_test)

model_xgb.score(x_train, y_train)

"""valide edilmiş (doğrulanmış hata oranımızı buluyoruz)"""

np.sqrt(-1*(cross_val_score(model_xgb, x_test, y_test, cv=10, scoring='neg_mean_squared_error'))).mean()