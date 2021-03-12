import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib.font_manager import FontProperties
import japanize_matplotlib
import seaborn as sns
from scipy.optimize import curve_fit
import itertools
#C2H2 O2
#a1son=[147.801 ,186.355 ,527.103 ,133.262 ,263.363 ]
#a1=[157.16193482,195.24380405,506.0792812,143.99351845,260.85222752]
a1=[147.801 ,186.355 ,527.103 ,133.262 ,263.363 ]
fai1=[1.15 ,1.0 ,0.55 ,1.3 , 0.8]
#C2H4 O2
#a2son=[784.015 ,546.57 ,542.52 ,1401.14 ,548.41 ,557.21 ]
#a2=[760.1702947566895,560.7173620342139,557.632863713739,1331.0704321449603,551.8857978467067,598.7164116111799]
a2=[784.015 ,546.57 ,542.52 ,1401.14 ,548.41 ,557.21 ]
fai2=[0.89 ,1.27 ,1.0 ,0.55 ,1.45,1.6 ]

#H2 O2
#a3son=[2253.611 ,2224.94 ,2283.07 ,2440.47 ,2285.95 ,2977.94 ,2395.81 ]
#a3=[2238.377993462328,2291.834342925457,2250.81678701269,2389.840030504452,2250.4642907522584,3048.3608072668276,2319.3691956706457,]
a3=[2253.611 ,2224.94 ,2283.07 ,2440.47 ,2285.95 ,2977.94 ,2395.81, 2642.705]
fai3=[1.0 ,0.83 ,0.95 ,1.05 ,0.63 ,1.3 ,0.55, 1.15]

#C2H6

a4=[1248.2786894978278,1882.8530868246514,1296.9292278022342,1267.6449453938626]
fai4=[1.0 ,0.7 ,1.39 ,1.2 ]

#C2H2AR50
a5=[454.7146045]
fai5=[1]

#C2H2AR80
a6=[1955.638]
fai6=[1]

#C2H4AR75
a7=[2894.671433]
fai7=[1]


def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))


df = pd.read_excel('\\CEA\\c2h2-c2h4-h2-c2h6-C2H2AR50-C2H2AR80-C2H4AR75.xlsx')

pc=seikika(df['pc[bar]'])
df['pc[bar]']=pc
Mc=seikika(df['Mc'])
df['Mc']=Mc
gammac=seikika(df['gamma_c'])
df['gamma_c']=gammac
p=seikika(df['p[bar]'])
df['p[bar]']=p
T=seikika(df['T[K]'])
df['T[K]']=T
M=seikika(df['M'])
df['M']=M
gamma=seikika(df['gamma'])
df['gamma']=gamma
VCJ=seikika(df['V_CJ[m/s]'])
df['V_CJ[m/s]']=VCJ

LR=df['反射点距離']

#l=['pc[bar]','Tc[K]', 'Hc[KJ/kg]', 'Mc', 'gamma_c', 'SonicVelocity_c[m/s]','p[bar]', 'T[K]', 'rho[kg/m^3]', 'H[KJ/kg]', 'U[KJ/kg]', 'G[KJ/kg]', 'S[KJ/kg K]',\
#                    'M', '(dLV/dLP)t', '(dLV/dLT)p', 'Cp[KJ/kg K]', 'gamma', 'SonicVelocity[m/s]','p/pc', 'T/Tc', 'M/Mc', 'rho/rho_c', 'M_CJ', 'V_CJ[m/s]']

l=['pc[bar]', 'Mc', 'gamma_c','p[bar]', 'T[K]',\
                    'M', 'gamma','V_CJ[m/s]']


d = {'物理量1':['1'] ,'物理量2':['i'] ,'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)

l1=list(itertools.combinations(l,2))
Amin=10000
Bmax=-10
print(len(l1))

for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    d = {ll[0]:aa ,ll[1]:bb ,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
print(Amin,C,D)
print(Bmax,C1,D1)

dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params2.xlsx')



d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)

l1=list(itertools.combinations(l,3))
Amin=10000
Bmax=-10
print(len(l1))

for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    cc=df[ll[2]]
    d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],ll[2],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
        E=ll[2]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
        E1=ll[2]
print(Amin,C,D,E)
print(Bmax,C1,D1,E1)

dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params3.xlsx')


l1=list(itertools.combinations(l,4))
Amin=10000
Bmax=-10
print(len(l1))
d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)

for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    cc=df[ll[2]]
    dd=df[ll[3]]
    d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
        E=ll[2]
        F=ll[3]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
        E1=ll[2]
        F1=ll[3]
print(Amin,C,D,E,F)
print(Bmax,C1,D1,E1,F1)

dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params4.xlsx')


l1=list(itertools.combinations(l,5))
Amin=10000
Bmax=-10
print(len(l1))

d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'],'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)
for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    cc=df[ll[2]]
    dd=df[ll[3]]
    ee=df[ll[4]]
    d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
        E=ll[2]
        F=ll[3]
        G=ll[4]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
        E1=ll[2]
        F1=ll[3]
        G1=ll[4]
print(Amin,C,D,E,F,G)
print(Bmax,C1,D1,E1,F1,G1)

dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params5.xlsx')

print(len(l1))

l1=list(itertools.combinations(l,6))
Amin=10000
Bmax=-10

d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'],'物理量6':['1'],'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)

for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    cc=df[ll[2]]
    dd=df[ll[3]]
    ee=df[ll[4]]
    ff=df[ll[5]]
    d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,ll[5]:ff,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],ll[5],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
        E=ll[2]
        F=ll[3]
        G=ll[4]
        H=ll[5]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
        E1=ll[2]
        F1=ll[3]
        G1=ll[4]
        H1=ll[5]
print(Amin,C,D,E,F,G,H)
print(Bmax,C1,D1,E1,F1,G1,H1)


dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params6.xlsx')

Amin=10000
Bmax=-10

l1=list(itertools.combinations(l,7))
print(len(l1))
d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'],'物理量6':['1'],'物理量7':['1'],'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)
for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    cc=df[ll[2]]
    dd=df[ll[3]]
    ee=df[ll[4]]
    ff=df[ll[5]]
    gg=df[ll[6]]
    d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,ll[5]:ff,ll[6]:gg,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],ll[5],ll[6],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
        E=ll[2]
        F=ll[3]
        G=ll[4]
        H=ll[5]
        I=ll[6]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
        E1=ll[2]
        F1=ll[3]
        G1=ll[4]
        H1=ll[5]
        I1=ll[6]
print(Amin,C,D,E,F,G,H,I)
print(Bmax,C1,D1,E1,F1,G1,H1,I1)
Amin=10000
Bmax=-10
dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params7.xlsx')

d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'],'物理量6':['1'],'物理量7':['1'],'物理量8':['1'],'MSE':[10000],'R2':[0]}

dfss = pd.DataFrame(d)
l1=list(itertools.combinations(l,8))
print(len(l1))
for i in range(len(l1)):
    ll=l1[i]
    aa=df[ll[0]]
    bb=df[ll[1]]
    cc=df[ll[2]]
    dd=df[ll[3]]
    ee=df[ll[4]]
    ff=df[ll[5]]
    gg=df[ll[6]]
    hh=df[ll[7]]
    d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,ll[5]:ff,ll[6]:gg,ll[7]:hh,'反射点距離':LR}
    dfs = pd.DataFrame(d)
    x = dfs.drop("反射点距離", axis=1)
    y = dfs['反射点距離']
    x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
    models = MLPRegressor(hidden_layer_sizes=(80,80,80,80,80,80),alpha=1e+0,solver='adam',activation='relu')
    models.fit(x_trains, y_trains)
    results=models.predict(x_tests)
    A=mean_squared_error(y_tests, results)
    B=r2_score(y_tests,results)
    dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],ll[5],ll[6],ll[7],A,B]
    if A <= Amin:
        Amin=A
        C=ll[0]
        D=ll[1]
        E=ll[2]
        F=ll[3]
        G=ll[4]
        H=ll[5]
        I=ll[6]
        J=ll[7]
    if B > Bmax:
        Bmax=B
        C1=ll[0]
        D1=ll[1]
        E1=ll[2]
        F1=ll[3]
        G1=ll[4]
        H1=ll[5]
        I1=ll[6]
        J1=ll[7]
print(Amin,C,D,E,F,G,H,I,J)
print(Bmax,C1,D1,E1,F1,G1,H1,I1,J1)

dfss=dfss.sort_values('R2', ascending=False)
dfss.to_excel('\\CEA\\params8.xlsx')

