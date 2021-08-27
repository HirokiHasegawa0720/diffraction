import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # no UI backend
import math
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor

'''
df = pd.read_excel('/mnt/c/CEA/all_mixturechem359.xlsx')

#3.1　特徴量の削減
l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'a0[m/s]', \
        'pcj[bar]', 'Tcj[K]', 'Hcj[KJ/kg]',\
                    'Mcj[kg/kmol]', 'γcj[-]', 'acj[m/s]',\
                         'Mcj[-]', 'Vcj[m/s]',\
        'Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', \
                      'γvn[-]','avn[m/s]',\
                    'inductionlength[m]', 'reactionlength[m]', 'Ea[KJ/kg]','LR']

dfs = pd.DataFrame({'p0[bar]':df[l[0]]})

for i in range(len(l)):
    params=df[l[i]]
    dfs[l[i]]=params

pulse=[]
params=dfs['inductionlength[m]']

for i in range(len(dfs)):
    if params[i]==0:
        pulse.append(i)
    else:
        pass

    
dfs.drop(index=dfs.index[pulse])

x = dfs

threshold = 0.9

feat_corr = set()
corr_matrix = x.corr()
corr_matrix.to_excel('/mnt/c/CEA/matrix1.xlsx')
plt.figure()
sns.heatmap(corr_matrix,cmap='viridis')
plt.tight_layout()
plt.savefig('gurafu0(研究報告).png')
plt.close('all')

for i in range(len(corr_matrix.columns)):
    z=0
    corr_matrix.iloc[i, i]=0
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feat_name1 = corr_matrix.columns[i]
            feat_name2 = corr_matrix.columns[j]
            corr_matrix.iloc[i, j]=0
            corr_matrix.iloc[j,i]=0
            corr_matrix.iloc[i, z]=feat_name2
            feat_corr.add(feat_name1)
            z=z+1

        else:
            corr_matrix.iloc[i, j]=0
            corr_matrix.iloc[j,i]=0
        
x.drop(labels=feat_corr, axis='columns', inplace=True)
print(x.columns)
print(len(x.columns))

corr_matrix.to_excel('/mnt/c/CEA/matrix2.xlsx')
'''


#3.2 グリッドサーチの結果

#MSE vs サーチケース
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

def plot_feature_importance(df):
    n_features = len(df)
    df_plot = df.sort_values('importance')
    f_importance_plot = df_plot['importance'].values
    plt.barh(range(n_features), f_importance_plot, align='center')
    cols_plot = df_plot['feature'].values
    plt.yticks(np.arange(n_features), cols_plot)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')

#reaction length
df = pd.read_excel('/mnt/c/CEA/all_mixturechem359.xlsx')

l1=['Fuel','Diluent','Equivalentratio','Diluentratio','p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Hvn[KJ/kg]',
       'inductionlength[m]', 'reactionlength[m]', 'Ea[KJ/kg]','LR']

XpcaFLRDED = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l1)):
    XpcaFLRDED[l1[i]]=df[l1[i]]

XpcaFLRDED[l1[i]]=df[l1[i]]

def calc_double(n):
    return n * 1000000

induction1=df['inductionlength[m]']

# data1の要素を全て倍にしてdata2に格納する
induction2 = list(map(calc_double, induction1))

XpcaFLRDED['inductionlength[m]'] = induction2

pulse=[]
params=XpcaFLRDED['inductionlength[m]']

for i in range(len(XpcaFLRDED)):
    if params[i]==0:
        pulse.append(i)
    else:
        pass

XpcaFLRDED=XpcaFLRDED.drop(index=pulse)

Xpca = XpcaFLRDED.drop(columns=['LR','Diluent','Fuel','Equivalentratio','Diluentratio'])

Diluent=list(XpcaFLRDED['Diluent'])
Fuel=list(XpcaFLRDED['Fuel'])
Equivalentratio=list(XpcaFLRDED['Equivalentratio'])
Diluentratio=list(XpcaFLRDED['Diluentratio'])


###1
dfH2train =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == 'H2') | ~( XpcaFLRDED['Equivalentratio'] == 1)]
dfH2test = XpcaFLRDED[(XpcaFLRDED['Fuel'] == 'H2') & (XpcaFLRDED['Equivalentratio'] == 1)]

dfC2H2train =  dfH2train[~( dfH2train['Fuel'] == 'C2H2') | ~( dfH2train['Equivalentratio'] == 1) | ~(dfH2train['Diluent'] == 0)]
dfC2H2test =  dfH2train[(dfH2train['Fuel'] == 'C2H2') & (dfH2train['Equivalentratio'] == 1) & (dfH2train['Diluent'] == 0)]

dfC2H4train =  dfC2H2train[~( dfC2H2train['Fuel'] == 'C2H4') | ~( dfC2H2train['Equivalentratio'] == 1) | ~(dfC2H2train['Diluent'] == 0)]
dfC2H4test =  dfC2H2train[(dfC2H2train['Fuel'] == 'C2H4') & (dfC2H2train['Equivalentratio'] == 1) & (dfC2H2train['Diluent'] == 0)]

dfC2H6train =  dfC2H4train[~( dfC2H4train['Fuel'] == 'C2H6') | ~(dfC2H4train['Equivalentratio'] == 1) | ~(dfC2H4train['Diluentratio'] == 0)]
dfC2H6test =  dfC2H4train[(dfC2H4train['Fuel'] == 'C2H6') & (dfC2H4train['Equivalentratio'] == 1) & (dfC2H4train['Diluentratio'] == 0)]

dfC2H2ARtrain =  dfC2H6train[~( dfC2H6train['Fuel'] == 'C2H2') | ~( dfC2H6train['Diluent'] == 'Ar') | ~(dfC2H6train['Diluentratio'] == 50)]
dfC2H2ARtest = dfC2H6train[(dfC2H6train['Fuel'] == 'C2H2') & (dfC2H6train['Diluent'] == 'Ar') & (dfC2H6train['Diluentratio'] == 50)]

dfC2H2N2train =  dfC2H2ARtrain[~( dfC2H2ARtrain['Fuel'] == 'C2H2') | ~( dfC2H2ARtrain['Diluent'] == 'N2') | ~(dfC2H2ARtrain['Diluentratio'] == 50)]
dfC2H2N2test = dfC2H2ARtrain[(dfC2H2ARtrain['Fuel'] == 'C2H2') & (dfC2H2ARtrain['Diluent'] == 'N2') & (dfC2H2ARtrain['Diluentratio'] == 50)]

dfC2H4ARtrain =  dfC2H2N2train[~( dfC2H2N2train['Fuel'] == 'C2H4') | ~( dfC2H2N2train['Diluent'] == 'Ar') | ~(dfC2H2N2train['Diluentratio'] == 50)]
dfC2H4ARtest = dfC2H2N2train[(dfC2H2N2train['Fuel'] == 'C2H4') & (dfC2H2N2train['Diluent'] == 'Ar') & (dfC2H2N2train['Diluentratio'] == 50)]

dfC2H4ARtrain = dfC2H4ARtrain.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfH2test = dfH2test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H2test = dfC2H2test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H4test = dfC2H4test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H6test = dfC2H6test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H2ARtest = dfC2H2ARtest.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H2N2test = dfC2H2N2test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H4ARtest  = dfC2H4ARtest .drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])

XH2test = dfH2test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
XC2H2test = dfC2H2test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
XC2H4test = dfC2H4test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
XC2H6test = dfC2H6test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
XC2H2ARtest = dfC2H2ARtest.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
XC2H2N2test = dfC2H2N2test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
XC2H4ARtest = dfC2H4ARtest.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])

yH2test = dfH2test['LR']
yC2H2test = dfC2H2test['LR']
yC2H4test = dfC2H4test['LR']
yC2H6test = dfC2H6test['LR']
yC2H2ARtest = dfC2H2ARtest['LR']
yC2H2N2test = dfC2H2N2test['LR']
yC2H4ARtest = dfC2H4ARtest['LR']

X_val= dfC2H4ARtrain.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
y_val =  dfC2H4ARtrain['LR']

X_train, X_test, y_train, y_test = train_test_split(X_val ,y_val, train_size=0.9, random_state=123)

ss = StandardScaler()
sX_train = ss.fit_transform(X_train)
sX_test = ss.transform(X_test)
sXH2test = ss.transform(XH2test)
sXC2H2test = ss.transform(XC2H2test)
sXC2H4test = ss.transform(XC2H4test)
sXC2H6test = ss.transform(XC2H6test)
sXC2H2ARtest = ss.transform(XC2H2ARtest)
sXC2H2N2test = ss.transform(XC2H2N2test)
sXC2H4ARtest = ss.transform(XC2H4ARtest)
sX_train = pd.DataFrame(sX_train,columns=X_train.columns)

sol=['adam']
act=['relu']
hidd=[]

for i in [2,4,6]:
    for j in [100,150,200]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)

alp=[1e-4,1e-2,1e+0]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}

grid1=GridSearchCV(MLPRegressor(), param_grid , cv=5 , n_jobs=4)
grid1.fit(sX_train,y_train)
result1mlp1=grid1.predict(sX_test)
MSE1=mean_squared_error(y_test, result1mlp1)
R1=r2_score(y_test,result1mlp1)
print('Test set score: {}'.format(grid1.score(sX_test, y_test)))
print('Best parameters: {}'.format(grid1.best_params_))
print('Best cross-validation: {}'.format(grid1.best_score_))

resultH2=grid1.predict(sXH2test)
MSEH2=mean_squared_error(yH2test, resultH2)
R2H2=r2_score(yH2test,resultH2)

resultC2H2=grid1.predict(sXC2H2test)
MSEC2H2=mean_squared_error(yC2H2test, resultC2H2)
R2C2H2=r2_score(yC2H2test,resultC2H2)

print(sXC2H4test)
resultC2H4=grid1.predict(sXC2H4test)
MSEC2H4=mean_squared_error(yC2H4test, resultC2H4)
R2C2H4=r2_score(yC2H4test,resultC2H4)

resultC2H6=grid1.predict(sXC2H6test)
MSEC2H6=mean_squared_error(yC2H6test, resultC2H6)
R2C2H6=r2_score(yC2H6test,resultC2H6)

resultC2H2AR=grid1.predict(sXC2H2ARtest)
MSEC2H2AR=mean_squared_error(yC2H2ARtest, resultC2H2AR)
R2C2H2AR=r2_score(yC2H2ARtest,resultC2H2AR)

resultC2H2N2=grid1.predict(sXC2H2N2test)
MSEC2H2N2=mean_squared_error(yC2H2N2test, resultC2H2N2)
R2C2H2N2=r2_score(yC2H2N2test,resultC2H2N2)

resultC2H4AR=grid1.predict(sXC2H4ARtest)
MSEC2H4AR=mean_squared_error(yC2H4ARtest, resultC2H4AR)
R2C2H4AR=r2_score(yC2H4ARtest,resultC2H4AR)

print(pd.DataFrame([[MSEH2,R2H2], [MSEC2H2,R2C2H2], [MSEC2H4, R2C2H4],[MSEC2H6, R2C2H6],[MSEC2H2AR, R2C2H2AR],[MSEC2H2N2, R2C2H2N2],[MSEC2H4AR, R2C2H4AR]],
                   columns=['MSE', 'R2'],
                   index=['H2', 'C2H2', 'C2H4','C2H6','C2H2AR','C2H2N2','C2H4AR']))


'''
for i in range(len(fuels)):
    fu1=fuels[i]
    eq=1

    dfH2train =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == fu1) | ~( XpcaFLRDED['Equivalentratio'] == eq)]
    dfH2test = XpcaFLRDED[(XpcaFLRDED['Fuel'] == fu1) & (XpcaFLRDED['Equivalentratio'] == eq)]
    dfH2train = dfH2train.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
    dfH2test = dfH2test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
    XH2test = dfH2test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
    yH2test = dfH2test['inductionlength[m]']

    X_valH2 = dfH2train.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
    y_valH2 =  dfH2train['inductionlength[m]']

    X_trainH2, X_testH2, y_trainH2, y_testH2 = train_test_split(X_valH2 ,y_valH2, train_size=0.6, random_state=123)

    ss = StandardScaler()

    sX_trainH2 = ss.fit_transform(X_trainH2)
    sX_testH2 = ss.transform(X_testH2)
    sXH2test = ss.transform(XH2test)
    sX_trainH2 = pd.DataFrame(sX_trainH2,columns=X_trainH2.columns)

    sol=['adam']
    act=['relu']
    hidd=[]

    for i in [2,3,4]:
        for j in [10,50,100]:
            b=[j]*i
            b=tuple(b)
            hidd.append(b)

    alp=[1e-4,1e-2,1e+0]
    param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}

    grid1=GridSearchCV(MLPRegressor(), param_grid , cv=2 , n_jobs=4)
    grid1.fit(sX_trainH2,y_trainH2)
    result1mlp1=grid1.predict(sX_testH2)
    MSE1=mean_squared_error(y_testH2, result1mlp1)
    R1=r2_score(y_testH2,result1mlp1)
    print('Test set score: {}'.format(grid1.score(sX_testH2, y_testH2)))
    print('Best parameters: {}'.format(grid1.best_params_))
    print('Best cross-validation: {}'.format(grid1.best_score_))
    result2mlp1=grid1.predict(sXH2test)
    MSE2=mean_squared_error(yH2test, result2mlp1)
    R2=r2_score(yH2test,result2mlp1)
    FUEL.append(fuels[i])
    DILUENT.append(diluents[i])
    MSE.append(MSE2)
    R.append(R2)


fuel1s=['C2H2','C2H2','C2H4']
diluents=['Ar','N2','Ar']

for i in range(len(fuel1s)):

    fu=fuel1s[i]
    diru=50
    di= diluents[i]

    dfC2H4ARtrain =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == fu) | ~( XpcaFLRDED['Diluentratio'] == diru) | ~(XpcaFLRDED['Diluent'] == di)]
    dfC2H4ARtest = XpcaFLRDED[(XpcaFLRDED['Fuel'] == fu) & (XpcaFLRDED['Diluentratio'] == diru) & (XpcaFLRDED['Diluent'] == di)]
    dfC2H4ARtrain = dfC2H4ARtrain.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
    dfC2H4ARtest = dfC2H4ARtest.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
    XC2H4ARtest = dfC2H4ARtest.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
    yC2H4ARtest = dfC2H4ARtest['inductionlength[m]']

    X_valC2H4AR = dfC2H4ARtrain.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
    y_valC2H4AR =  dfC2H4ARtrain['inductionlength[m]']

    X_trainH2, X_testH2, y_trainH2, y_testH2 = train_test_split(X_valH2 ,y_valH2, train_size=0.6, random_state=123)
    X_trainC2H4AR, X_testC2H4AR, y_trainC2H4AR, y_testC2H4AR = train_test_split(X_valC2H4AR ,y_valC2H4AR, train_size=0.6, random_state=123)

    sX_trainC2H4AR = ss.fit_transform(X_trainC2H4AR)
    sX_testC2H4AR = ss.transform(X_testC2H4AR)
    sXC2H4ARtest = ss.transform(XC2H4ARtest)
    sXC2H4ARtest = pd.DataFrame(sXC2H4ARtest,columns=X_trainC2H4AR.columns)

    #MSE vs エポック

    grid2=GridSearchCV(MLPRegressor(), param_grid , cv=2 , n_jobs=4)
    grid2.fit(sX_trainC2H4AR,y_trainC2H4AR)
    result1mlp2=grid2.predict(sX_testC2H4AR)
    MSE3=mean_squared_error(y_testC2H4AR, result1mlp2)
    R3=r2_score(y_testC2H4AR,result1mlp2)
    print('Test set score: {}'.format(grid2.score(sX_testC2H4AR, y_testC2H4AR)))
    print('Best parameters: {}'.format(grid2.best_params_))
    print('Best cross-validation: {}'.format(grid2.best_score_))
    result2mlp2=grid2.predict(sXC2H4ARtest)
    MSE4=mean_squared_error(yC2H4ARtest, result2mlp2)
    R4=r2_score(yC2H4ARtest,result2mlp2)

    FUEL.append(fuel1s[i])
    DILUENT.append(diluents[i])
    MSE.append(MSE4)
    R.append(R4)


gaibu = pd.DataFrame({'fuel':FUEL})

gaibu['diluent']=DILUENT
gaibu['MSE']=MSE
gaibu['R2']=R


gaibu.to_excel('/mnt/c/CEA/validation.xlsx')
'''