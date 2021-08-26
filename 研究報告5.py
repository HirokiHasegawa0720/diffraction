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

print(induction2)

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

for i in range(len())
fu1='C2H4'
eq=1

dfH2train =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == fu1) | ~( XpcaFLRDED['Equivalentratio'] == eq)]
dfH2test = XpcaFLRDED[(XpcaFLRDED['Fuel'] == fu1) & (XpcaFLRDED['Equivalentratio'] == eq)]
dfH2train = dfH2train.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfH2test = dfH2test.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
XH2test = dfH2test.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
yH2test = dfH2test['inductionlength[m]']

fu='C2H4'
diru=50
di= 'Ar'

dfC2H4ARtrain =  XpcaFLRDED[~( XpcaFLRDED['Fuel'] == fu) | ~( XpcaFLRDED['Diluentratio'] == diru) | ~(XpcaFLRDED['Diluent'] == di)]
dfC2H4ARtest = XpcaFLRDED[(XpcaFLRDED['Fuel'] == fu) & (XpcaFLRDED['Diluentratio'] == diru) & (XpcaFLRDED['Diluent'] == di)]
dfC2H4ARtrain = dfC2H4ARtrain.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
dfC2H4ARtest = dfC2H4ARtest.drop(columns=['Diluent','Fuel','Equivalentratio','Diluentratio'])
XC2H4ARtest = dfC2H4ARtest.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
yC2H4ARtest = dfC2H4ARtest['inductionlength[m]']
print(len(dfC2H4ARtest))
print(len(dfC2H4ARtrain))

X_valH2 = dfH2train.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
y_valH2 =  dfH2train['inductionlength[m]']
X_valC2H4AR = dfC2H4ARtrain.drop(columns=['LR','inductionlength[m]','reactionlength[m]'])
y_valC2H4AR =  dfC2H4ARtrain['inductionlength[m]']

X_trainH2, X_testH2, y_trainH2, y_testH2 = train_test_split(X_valH2 ,y_valH2, train_size=0.6, random_state=123)
X_trainC2H4AR, X_testC2H4AR, y_trainC2H4AR, y_testC2H4AR = train_test_split(X_valC2H4AR ,y_valC2H4AR, train_size=0.6, random_state=123)

print(X_trainH2.head())

ss = StandardScaler()

sX_trainH2 = ss.fit_transform(X_trainH2)
sX_testH2 = ss.transform(X_testH2)
sXH2test = ss.transform(XH2test)
sX_trainH2 = pd.DataFrame(sX_trainH2,columns=X_trainH2.columns)

sX_trainC2H4AR = ss.fit_transform(X_trainC2H4AR)
sX_testC2H4AR = ss.transform(X_testC2H4AR)
sXC2H4ARtest = ss.transform(XC2H4ARtest)
sXC2H4ARtest = pd.DataFrame(sXC2H4ARtest,columns=X_trainC2H4AR.columns)

sXpca = ss.fit_transform(Xpca)

#MSE vs エポック

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

print(MSE1,R1,'mlp',fu1,eq)
print(MSE2,R2,'mlp',fu1,eq)
print(MSE3,R3,'mlp',fu,diru,di)
print(MSE4,R4,'mlp',fu,diru,di)






'''
from sklearn import svm

model = svm.SVR(C=1.0, kernel='rbf', epsilon=0.1)
model.fit(sX_trainH2,y_trainH2)
result1svr=model.predict(sX_testH2)
MSE5=mean_squared_error(y_testH2, result1svr)
R5=r2_score(y_testH2, result1svr)
result2svr = model.predict(sXH2test)
MSE6=mean_squared_error(yH2test, result2svr)
R6=r2_score(yH2test, result2svr)
model = svm.SVR(C=1.0, kernel='rbf', epsilon=0.1)
model.fit(sX_trainC2H4AR,y_trainC2H4AR)
result1svr=model.predict(sX_testC2H4AR)
MSE7=mean_squared_error(y_testC2H4AR, result1svr)
R7=r2_score(y_testC2H4AR,result1svr)
result2svr=model.predict(sXC2H4ARtest)
MSE8=mean_squared_error(yC2H4ARtest, result2svr)
R8=r2_score(yC2H4ARtest,result2svr)
print(MSE5,R5,'svm',fu1,eq)
print(MSE6,R6,'svm',fu1,eq)
print(MSE7,R7,'svm',fu,diru,di)
print(MSE8,R8,'svm',fu,diru,di)

#3.3 あてはめ性能 
#Ypredict vs Yexp(学習データ)
#mlpregressor

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

values =  np.concatenate([y_testH2, result1mlp1], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1


plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_testH2,result1mlp1,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.title("mlp")
plt.tight_layout()
plt.savefig("gurafu4(研究報告).png")


#Ypredict vs Yexp(外部テストデータ)


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


values =  np.concatenate([yH2test, result2mlp1], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(yH2test,result2mlp1,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.title("mlp")
#plt.xticks(np.arange(6, 11, step=2))
#plt.yticks(np.arange(6, 11, step=2))
plt.tight_layout()
plt.savefig("gurafu5(研究報告).png")

#3.4特徴量の寄与度

#順列重要度vs特徴量


result = permutation_importance(grid1,sX_trainH2,y_trainH2, n_repeats=5, random_state=42)

cols = list(sX_trainH2.columns)         # 特徴量名のリスト(目的変数CRIM以外)
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
plt.savefig("gurafu6(研究報告).png")



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

values =  np.concatenate([y_testC2H4AR, result1mlp1], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1


plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_testC2H4AR,result1mlp2,s=30)
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.title("mlp")
plt.tight_layout()
plt.savefig("gurafu7(研究報告).png")


#Ypredict vs Yexp(外部テストデータ)


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


values =  np.concatenate([yC2H4ARtest, result2mlp2], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(yC2H4ARtest,result2mlp2,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)
#plt.xticks(np.arange(5, 20, step=5))
#plt.yticks(np.arange(5, 20, step=5))
plt.xlabel('LR experiment [mm]')
plt.ylabel('LR predict [mm]')
plt.title("mlp")
plt.tight_layout()
plt.savefig("gurafu8(研究報告).png")

#3.4特徴量の寄与度

#順列重要度vs特徴量


result = permutation_importance(grid2,sX_trainC2H4AR,y_trainC2H4AR, n_repeats=5, random_state=42)

cols = list(sX_trainH2.columns)         # 特徴量名のリスト(目的変数CRIM以外)
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
plt.savefig("gurafu9(研究報告).png")
'''