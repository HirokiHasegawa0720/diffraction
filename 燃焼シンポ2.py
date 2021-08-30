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

df = pd.read_excel('/mnt/c/CEA/all_mixtureexp.xlsx')

#3.1　特徴量の削減
l=['P0', 'T0','H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'a0[m/s]', \
        'pcj[bar]', 'Tcj[K]', 'Hcj[KJ/kg]',\
                    'Mcj[kg/kmol]', 'γcj[-]', 'acj[m/s]',\
                         'Mcj[-]', 'Vcj[m/s]',\
        'Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', \
                      'γvn[-]','avn[m/s]',\
                    'Lr']

dfs = pd.DataFrame({'P0[bar]':df[l[0]]})

for i in range(len(l)):
    params=df[l[i]]
    dfs[l[i]]=params

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
df = pd.read_excel('/mnt/c/CEA/all_mixtureexp.xlsx')

l1=['Fuel','Diluent','Equivalentratio','CoefficientDiluent','P0', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Hvn[KJ/kg]',
       'Lr']

XpcaFLRDED = pd.DataFrame({'P0':df['P0']})

for i in range(len(l1)):
    XpcaFLRDED[l1[i]]=df[l1[i]]

dfH2train =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == 'H2') | ~( XpcaFLRDED['Equivalentratio'] == 1)]
dfH2test = XpcaFLRDED[(XpcaFLRDED['Fuel'] == 'H2') & (XpcaFLRDED['Equivalentratio'] == 1)]

dfC2H2train =  dfH2train[~( dfH2train['Fuel'] == 'C2H2,acetylene') | ~( dfH2train['Equivalentratio'] == 1) | ~(dfH2train['Diluent'].isnull())]
dfC2H2test =  dfH2train[(dfH2train['Fuel'] == 'C2H2,acetylene') & (dfH2train['Equivalentratio'] == 1) & (dfH2train['Diluent'].isnull())]

dfC2H4train =  dfC2H2train[~( dfC2H2train['Fuel'] == 'C2H4') | ~( dfC2H2train['Equivalentratio'] == 1) | ~(dfC2H2train['Diluent'].isnull())]
dfC2H4test =  dfC2H2train[(dfC2H2train['Fuel'] == 'C2H4') & (dfC2H2train['Equivalentratio'] == 1) & (dfC2H2train['Diluent'].isnull())]

dfC2H6train =  dfC2H4train[~( dfC2H4train['Fuel'] == 'C2H6') | ~(dfC2H4train['Equivalentratio'] == 1) | ~(dfC2H4train['Diluent'].isnull())]
dfC2H6test =  dfC2H4train[(dfC2H4train['Fuel'] == 'C2H6') & (dfC2H4train['Equivalentratio'] == 1) & (dfC2H4train['Diluent'].isnull())]

dfC2H2ARtrain =  dfC2H6train[~( dfC2H6train['Fuel'] == 'C2H2,acetylene') | ~( dfC2H6train['Diluent'] == 'Ar') | ~(dfC2H6train['CoefficientDiluent'] == 7)]
dfC2H2ARtest = dfC2H6train[(dfC2H6train['Fuel'] == 'C2H2,acetylene') & (dfC2H6train['Diluent'] == 'Ar') & (dfC2H6train['CoefficientDiluent'] == 7)]

dfC2H2N2train =  dfC2H2ARtrain[~( dfC2H2ARtrain['Fuel'] == 'C2H2,acetylene') | ~( dfC2H2ARtrain['Diluent'] == 'N2') | ~(dfC2H2ARtrain['CoefficientDiluent'] == 7)]
dfC2H2N2test = dfC2H2ARtrain[(dfC2H2ARtrain['Fuel'] == 'C2H2,acetylene') & (dfC2H2ARtrain['Diluent'] == 'N2') & (dfC2H2ARtrain['CoefficientDiluent'] == 7)]

dfC2H4ARtrain =  dfC2H2N2train[~( dfC2H2N2train['Fuel'] == 'C2H4') | ~( dfC2H2N2train['Diluent'] == 'Ar') | ~(dfC2H2N2train['CoefficientDiluent'] == 4)]
dfC2H4ARtest = dfC2H2N2train[(dfC2H2N2train['Fuel'] == 'C2H4') & (dfC2H2N2train['Diluent'] == 'Ar') & (dfC2H2N2train['CoefficientDiluent'] == 4)]


dfC2H4ARtrain = dfC2H4ARtrain.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfH2test = dfH2test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfC2H2test = dfC2H2test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfC2H4test = dfC2H4test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfC2H6test = dfC2H6test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfC2H2ARtest = dfC2H2ARtest.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfC2H2N2test = dfC2H2N2test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])
dfC2H4ARtest  = dfC2H4ARtest .drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent'])

XH2test = dfH2test.drop(columns=['Lr'])
XC2H2test = dfC2H2test.drop(columns=['Lr'])
XC2H4test = dfC2H4test.drop(columns=['Lr'])
XC2H6test = dfC2H6test.drop(columns=['Lr'])
XC2H2ARtest = dfC2H2ARtest.drop(columns=['Lr'])
XC2H2N2test = dfC2H2N2test.drop(columns=['Lr'])
XC2H4ARtest = dfC2H4ARtest.drop(columns=['Lr'])

yH2test = dfH2test['Lr']
yC2H2test = dfC2H2test['Lr']
yC2H4test = dfC2H4test['Lr']
yC2H6test = dfC2H6test['Lr']
yC2H2ARtest = dfC2H2ARtest['Lr']
yC2H2N2test = dfC2H2N2test['Lr']
yC2H4ARtest = dfC2H4ARtest['Lr']

print(dfC2H4ARtrain)
dfC2H4ARtrain.to_excel('/mnt/c/CEA/bug.xlsx')
X_val= dfC2H4ARtrain.drop(columns=['Lr'])
y_val =  dfC2H4ARtrain['Lr']

X_train, X_test, y_train, y_test = train_test_split(X_val ,y_val, train_size=0.7, random_state=123)

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

for i in [3]:
    for j in [100]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)

alp=[1e-1,1e+1]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}

grid1=GridSearchCV(MLPRegressor(), param_grid , cv=2 , n_jobs=4)
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
print(len(XpcaFLRDED))
dfC2H4train =   XpcaFLRDED[~( XpcaFLRDED['Fuel'] == 'C2H4') | ~(  XpcaFLRDED['Equivalentratio'] == 1) | ~( XpcaFLRDED['Diluent'].isnull())]
dfC2H4test =   XpcaFLRDED[( XpcaFLRDED['Fuel'] == 'C2H4') & ( XpcaFLRDED['Equivalentratio'] == 1) & ( XpcaFLRDED['Diluent'].isnull())]
print(len(dfC2H4train))
print(len(dfC2H4test))

dfC2H4train = dfC2H4train.drop(columns=['Diluent','Fuel','Equivalentratio'])
dfC2H4test = dfC2H4test.drop(columns=['Diluent','Fuel','Equivalentratio'])

XC2H4test = dfC2H4test.drop(columns=['Lr'])

yC2H4test = dfC2H4test['Lr']

X_val= dfC2H4train.drop(columns=['Lr'])
y_val =  dfC2H4train['Lr']

X_train, X_test, y_train, y_test = train_test_split(X_val ,y_val, train_size=0.7, random_state=123)

ss = StandardScaler()
sX_train = ss.fit_transform(X_train)
sX_test = ss.transform(X_test)
sXC2H4test = ss.transform(XC2H4test)

sol=['adam']
act=['relu']
hidd=[]

for i in [3]:
    for j in [100]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)

alp=[1e+0]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}

grid1=GridSearchCV(MLPRegressor(), param_grid , cv=2 , n_jobs=4)
grid1.fit(sX_train,y_train)
result1mlp1=grid1.predict(sX_test)
MSE1=mean_squared_error(y_test, result1mlp1)
R1=r2_score(y_test,result1mlp1)
print('Test set score: {}'.format(grid1.score(sX_test, y_test)))
print('Best parameters: {}'.format(grid1.best_params_))
print('Best cross-validation: {}'.format(grid1.best_score_))


resultC2H4=grid1.predict(sXC2H4test)
MSEC2H4=mean_squared_error(yC2H4test, resultC2H4)
R2C2H4=r2_score(resultC2H4,yC2H4test)
print(MSEC2H4)
print(R2C2H4)
plt.figure(figsize=(5,5))

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 4.0

plt.rcParams['axes.linewidth'] = 2.0

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2.0

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.5

values =  np.concatenate([yC2H4test, resultC2H4], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1


plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(yC2H4test,resultC2H4,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.title("mlp")
plt.tight_layout()
plt.savefig("gurafu10(研究報告).png")
'''