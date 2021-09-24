from matplotlib import colors
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

df = pd.read_excel('all_mixtureexp15.xlsx')

#3.1　特徴量の削減
l=['P0', 'T0','H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'a0[m/s]', \
        'pcj[bar]', 'Tcj[K]', 'Hcj[KJ/kg]',\
                    'Mcj[kg/kmol]', 'γcj[-]', 'acj[m/s]',\
                         'Mcj[-]', 'Vcj[m/s]',\
        'Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', \
                      'γvn[-]','avn[m/s]','Ea[KJ/kg]']

dfs = pd.DataFrame({'P0':df[l[0]]})

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
plt.savefig('gurafu0.png')
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
df = pd.read_excel('all_mixtureexp15.xlsx')

l1=['Fuel','Diluent','Equivalentratio','CoefficientDiluent','Oxidizer','P0', 'T0', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Tvn[K]',
       'Ea[KJ/kg]',
       'Lr']

XpcaFLRDED = pd.DataFrame({'P0':df['P0']})

for i in range(len(l1)):
    XpcaFLRDED[l1[i]]=df[l1[i]]

XpcaFLRDED = XpcaFLRDED[~( XpcaFLRDED['Ea[KJ/kg]'] == 0)]

dfH2train =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == 'H2') | ~( XpcaFLRDED['Equivalentratio'] == 1)]
dfH2test = XpcaFLRDED[(XpcaFLRDED['Fuel'] == 'H2') & (XpcaFLRDED['Equivalentratio'] == 1)]

dftrainvalH2 = dfH2train[(dfH2train['Fuel'] == 'H2')]
print(len(dftrainvalH2),'H2')
print(len(dfH2test),'H2')

dfC2H2train =  dfH2train[~( dfH2train['Fuel'] == 'C2H2') | ~( dfH2train['Equivalentratio'] == 1) | ~(dfH2train['Diluent'].isnull())]
dfC2H2test =  dfH2train[(dfH2train['Fuel'] == 'C2H2') & (dfH2train['Equivalentratio'] == 1) & (dfH2train['Diluent'].isnull())]

dftrainvalC2H2 = dfC2H2train[(dfC2H2train['Fuel'] == 'C2H2') & (dfC2H2train['Diluent'].isnull())]
print(len(dftrainvalC2H2),'C2H2')
print(len(dfC2H2test),'C2H2')

dfC2H4train =  dfC2H2train[~( dfC2H2train['Fuel'] == 'C2H4') | ~( dfC2H2train['Equivalentratio'] == 1) | ~(dfC2H2train['Diluent'].isnull())]
dfC2H4test =  dfC2H2train[(dfC2H2train['Fuel'] == 'C2H4') & (dfC2H2train['Equivalentratio'] == 1) & (dfC2H2train['Diluent'].isnull())]

dftrainvalC2H4 = dfC2H4train[(dfC2H4train['Fuel'] == 'C2H4') & (dfC2H4train['Diluent'].isnull())]
print(len(dftrainvalC2H4),'C2H4')
print(len(dfC2H4test),'C2H4')

dfC2H6train =  dfC2H4train[~( dfC2H4train['Fuel'] == 'C2H6') | ~(dfC2H4train['Equivalentratio'] == 1) | ~(dfC2H4train['Diluent'].isnull()) | ~(dfC2H4train['Oxidizer'] == 'O2')]
dfC2H6test =  dfC2H4train[(dfC2H4train['Fuel'] == 'C2H6') & (dfC2H4train['Equivalentratio'] == 1) & (dfC2H4train['Diluent'].isnull()) & (dfC2H4train['Oxidizer'] == 'O2')]

dftrainvalC2H6 = dfC2H6train[(dfC2H6train['Fuel'] == 'C2H6')  & (dfC2H6train['Diluent'].isnull()) & (dfC2H6train['Oxidizer'] == 'O2')]
print(len(dftrainvalC2H6),'C2H6')
print(len(dfC2H6test),'C2H6')

dfC2H2ARtrain =  dfC2H6train[~( dfC2H6train['Fuel'] == 'C2H2') | ~( dfC2H6train['Diluent'] == 'Ar') | ~(dfC2H6train['CoefficientDiluent'] == 7)]
dfC2H2ARtest = dfC2H6train[(dfC2H6train['Fuel'] == 'C2H2') & (dfC2H6train['Diluent'] == 'Ar') & (dfC2H6train['CoefficientDiluent'] == 7)]

dftrainvalC2H2AR = dfC2H2ARtrain[(dfC2H2ARtrain['Fuel'] == 'C2H2') & (dfC2H2ARtrain['Diluent'] == 'Ar')]
print(len(dftrainvalC2H2AR),'C2H2AR')
print(len(dfC2H2ARtest),'C2H2AR')


dftrainval =  dfC2H2ARtrain[~( dfC2H2ARtrain['Fuel'] == 'C2H4') | ~( dfC2H2ARtrain['Diluent'] == 'Ar') | ~(dfC2H2ARtrain['CoefficientDiluent'] == 4)]
dfC2H4ARtest = dfC2H2ARtrain[(dfC2H2ARtrain['Fuel'] == 'C2H4') & (dfC2H2ARtrain['Diluent'] == 'Ar') & (dfC2H2ARtrain['CoefficientDiluent'] == 4)]

dftrainvalC2H4AR = dftrainval[(dftrainval['Fuel'] == 'C2H4') & (dftrainval['Diluent'] == 'Ar')]
print(len(dftrainvalC2H4AR),'C2H4AR')
print(len(dfC2H4ARtest),'C2H4AR')


dftrainvalC2H6N2O = dftrainval[(dftrainval['Fuel'] == 'C2H6') & (dftrainval['Oxidizer'] == 'N2O')]
print(len(dftrainvalC2H6N2O),'N2O')

dftrainvalC2H2N2 = dftrainval[(dftrainval['Fuel'] == 'C2H2') & (dftrainval['Diluent'] == 'N2')]
print(len(dftrainvalC2H2N2),'N2')

dftrainval = dftrainval.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dfH2test = dfH2test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dfC2H2test = dfC2H2test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dfC2H4test = dfC2H4test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dfC2H6test = dfC2H6test.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dfC2H2ARtest = dfC2H2ARtest.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dfC2H4ARtest  = dfC2H4ARtest.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])


dftrainvalH2 = dftrainvalH2.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H2 = dftrainvalC2H2.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H4 = dftrainvalC2H4.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H6 = dftrainvalC2H6.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H2AR = dftrainvalC2H2AR.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H4AR  = dftrainvalC2H4AR.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H6N2O = dftrainvalC2H6N2O.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])
dftrainvalC2H2N2 = dftrainvalC2H2N2.drop(columns=['Diluent','Fuel','Equivalentratio','CoefficientDiluent','Oxidizer'])


XH2test = dfH2test.drop(columns=['Lr'])
XC2H2test = dfC2H2test.drop(columns=['Lr'])
XC2H4test = dfC2H4test.drop(columns=['Lr'])
XC2H6test = dfC2H6test.drop(columns=['Lr'])
XC2H2ARtest = dfC2H2ARtest.drop(columns=['Lr'])
XC2H4ARtest = dfC2H4ARtest.drop(columns=['Lr'])

XtrainvalH2 = dftrainvalH2.drop(columns=['Lr'])
XtrainvalC2H2 = dftrainvalC2H2.drop(columns=['Lr'])
XtrainvalC2H4 = dftrainvalC2H4.drop(columns=['Lr'])
XtrainvalC2H6 = dftrainvalC2H6.drop(columns=['Lr'])
XtrainvalC2H2AR = dftrainvalC2H2AR.drop(columns=['Lr'])
XtrainvalC2H4AR = dftrainvalC2H4AR.drop(columns=['Lr'])
XtrainvalC2H6N2O = dftrainvalC2H6N2O.drop(columns=['Lr'])
XtrainvalC2H2N2 = dftrainvalC2H2N2.drop(columns=['Lr'])

yH2test = dfH2test['Lr']
yC2H2test = dfC2H2test['Lr']
yC2H4test = dfC2H4test['Lr']
yC2H6test = dfC2H6test['Lr']
yC2H2ARtest = dfC2H2ARtest['Lr']
yC2H4ARtest = dfC2H4ARtest['Lr']


ytrainvalH2 = dftrainvalH2['Lr']
ytrainvalC2H2 = dftrainvalC2H2['Lr']
ytrainvalC2H4 = dftrainvalC2H4['Lr']
ytrainvalC2H6 = dftrainvalC2H6['Lr']
ytrainvalC2H2AR = dftrainvalC2H2AR['Lr']
ytrainvalC2H4AR= dftrainvalC2H4AR['Lr']
ytrainvalC2H6N2O = dftrainvalC2H6N2O['Lr']
ytrainvalC2H2N2 = dftrainvalC2H2N2['Lr']


X_trainval= dftrainval.drop(columns=['Lr'])
y_trainval = dftrainval['Lr']

X_train, X_val, y_train, y_val = train_test_split(X_trainval ,y_trainval, train_size=0.7, random_state=123)

ss = StandardScaler()
sX_train = ss.fit_transform(X_train)
sX_val = ss.transform(X_val)
sXH2test = ss.transform(XH2test)
sXC2H2test = ss.transform(XC2H2test)
sXC2H4test = ss.transform(XC2H4test)
sXC2H6test = ss.transform(XC2H6test)
sXC2H2ARtest = ss.transform(XC2H2ARtest)
sXC2H4ARtest = ss.transform(XC2H4ARtest)
sX_train = pd.DataFrame(sX_train,columns=X_train.columns)

sXtrainvalH2 = ss.transform(XtrainvalH2)
sXtrainvalC2H2 = ss.transform(XtrainvalC2H2)
sXtrainvalC2H4 = ss.transform(XtrainvalC2H4)
sXtrainvalC2H6  = ss.transform(XtrainvalC2H6 )
sXtrainvalC2H2AR = ss.transform(XtrainvalC2H2AR)
sXtrainvalC2H4AR = ss.transform(XtrainvalC2H4AR )
sXtrainvalC2H6N2O = ss.transform(XtrainvalC2H6N2O)
sXtrainvalC2H2N2 = ss.transform(XtrainvalC2H2N2)

sol=['adam']
act=['relu']
hidd=[]

for i in [3,4]:
    for j in [100,150]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)

alp=[1e-4,1e-2]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}

grid1=GridSearchCV(MLPRegressor(), param_grid , cv=2 , n_jobs=4)
grid1.fit(sX_train,y_train)
result1=grid1.predict(sX_train)
MSE1=mean_squared_error(y_train, result1)
R1=r2_score(y_train, result1)

result2=grid1.predict(sX_val)
MSE2=mean_squared_error(y_val, result2)
R2=r2_score(y_val,result2)
print(MSE1,R1)
print(MSE2,R2)
print('Validation set score: {}'.format(grid1.score(sX_val, y_val)))
print('Best parameters: {}'.format(grid1.best_params_))
print('Best cross-validation: {}'.format(grid1.best_score_))

plt.figure()
cv_result = pd.DataFrame(grid1.cv_results_)
cv_result = cv_result[['param_hidden_layer_sizes', 'param_alpha', 'mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_hidden_layer_sizes', 'param_alpha')
print(cv_result_pivot )
heat_map = sns.heatmap(cv_result_pivot, cmap='viridis', annot=True)
plt.tight_layout()
plt.savefig("gurafu1.png")
plt.close('all')

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

resultC2H4AR=grid1.predict(sXC2H4ARtest)
MSEC2H4AR=mean_squared_error(yC2H4ARtest, resultC2H4AR)
R2C2H4AR=r2_score(yC2H4ARtest,resultC2H4AR)

print(pd.DataFrame([[MSEH2,R2H2,len(yH2test)], [MSEC2H2,R2C2H2,len(yC2H2test)], [MSEC2H4, R2C2H4,len(yC2H4test)],[MSEC2H6, R2C2H6,len(yC2H6test)],[MSEC2H2AR, R2C2H2AR,len(yC2H2ARtest)],[MSEC2H4AR, R2C2H4AR,len(yC2H4ARtest)]],
                   columns=['MSE', 'R2','len'],
                   index=['H2', 'C2H2', 'C2H4','C2H6','C2H2AR','C2H4AR']))



#training and validation score

resulttrainvalH2=grid1.predict(sXtrainvalH2)
resulttrainvalC2H2=grid1.predict(sXtrainvalC2H2)
resulttrainvalC2H4=grid1.predict(sXtrainvalC2H4)
resulttrainvalC2H6=grid1.predict(sXtrainvalC2H6)
resulttrainvalC2H2AR=grid1.predict(sXtrainvalC2H2AR)
resulttrainvalC2H4AR=grid1.predict(sXtrainvalC2H4AR)
resulttrainvalC2H6N2O=grid1.predict(sXtrainvalC2H6N2O)
resulttrainvalC2H2N2=grid1.predict(sXtrainvalC2H2N2)

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

plt.plot([0, 60], [0, 60],color='black')

plt.scatter(ytrainvalH2,resulttrainvalH2,s=10,color='grey')
plt.scatter(ytrainvalC2H2,resulttrainvalC2H2,s=10,color='red')
plt.scatter(ytrainvalC2H4,resulttrainvalC2H4,s=10,color='blue')
plt.scatter(ytrainvalC2H6,resulttrainvalC2H6,s=10,color='green')
plt.scatter(ytrainvalC2H2AR,resulttrainvalC2H2AR,s=10,color='black')
plt.scatter(ytrainvalC2H4AR,resulttrainvalC2H4AR,s=10,color='brown')
plt.scatter(ytrainvalC2H6N2O,resulttrainvalC2H6N2O,s=10,color='darkorange')
plt.scatter(ytrainvalC2H2N2,resulttrainvalC2H2N2,s=10,color='purple')


plt.xlim(0, 60)
plt.ylim(0, 60)
plt.xticks(np.arange(0, 61, step=20))
plt.yticks(np.arange(0, 61, step=20))
plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.tight_layout()
plt.savefig("gurafu2.png")


#Test score

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

plt.plot([0, 60], [0, 60],color='black')

plt.scatter(yH2test,resultH2,s=10,color='grey')
plt.scatter(yC2H2test,resultC2H2,s=10,color='red')
plt.scatter(yC2H4test,resultC2H4,s=10,color='blue')
plt.scatter(yC2H6test,resultC2H6,s=10,color='green')
plt.scatter(yC2H2ARtest,resultC2H2AR,s=10,color='black')
plt.scatter(yC2H4ARtest,resultC2H4AR,s=10,color='brown')

plt.xlim(0, 60)
plt.ylim(0, 60)
plt.xticks(np.arange(0, 61, step=20))
plt.yticks(np.arange(0, 61, step=20))
plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.tight_layout()
plt.savefig("gurafu3.png")


result = permutation_importance(grid1,sX_train,y_train, n_repeats=5, random_state=42)

cols = list(sX_train.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(result["importances"].mean(axis=1)) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df1=df_importance
df_importance = df_importance.sort_values("importance",ascending=False)

plt.figure(figsize=(8,8))

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 4.0

plt.rcParams['axes.linewidth'] = 2.0

# Tick Setting
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2.0

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5
#sns.barplot(x="Importance", y="Features",data=df_importance,ci=None)
plot_feature_importance(df_importance)
plt.title("Permutation Importance(mlp)")
plt.tight_layout()
plt.savefig("gurafu4.png")



'''
from sklearn.inspection import plot_partial_dependence

plt.figure(figsize=(5,10))


plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 4.0

plt.rcParams['axes.linewidth'] = 2.0

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams["xtick.bottom"] = False  
plt.rcParams['ytick.right'] = True

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2.0

plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.5


tree_disp = plot_partial_dependence(grid1, sX_train, ['Tcj[K]'])

plt.tight_layout()
plt.savefig("gurafu5(研究報告).png")
'''