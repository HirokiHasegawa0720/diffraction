from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
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


def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))

def plot_feature_importance(df):
    n_features = len(df)
    df_plot = df.sort_values('importance')
    f_importance_plot = df_plot['importance'].values
    plt.barh(range(n_features), f_importance_plot, align='center')
    cols_plot = df_plot['feature'].values
    plt.yticks(np.arange(n_features), cols_plot)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')


#33,34

n=10  #1つの混合気あたりのデータ点数
a=34  #C2H4/O2,AR50


def splitmixture(dfs,a,n):
        train = dfs.drop(range(a*n,(a+1)*n))
        test = dfs[a*n:(a+1)*n]
        return train,test



## allparameters
#######################################################


df = pd.read_excel('all_mixture.xlsx')


l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'a0[m/s]', \
        'pcj[bar]', 'Tcj[K]', 'Hcj[KJ/kg]',\
                    'Mcj[kg/kmol]', 'γcj[-]', 'acj[m/s]',\
                         'Mcj[-]', 'Vcj[m/s]',\
        'Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', \
                      'γvn[-]','avn[m/s]', \
                        ]

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params

#dfs=dfs.rename(columns={'p0[bar]': 'p0', 'H0[KJ/kg]': 'H0','M0kg/kmol': 'M0','gamma_0': 'γ0','p_CJ[bar]': 'pCJ', 'T_CJ[K]': 'TCJ', 'H_CJ[KJ/kg]': 'HCJ',  'M_CJkg/kmol': 'MCJ', 'gamma_CJ': 'γCJ', 'M_CJ': 'M_CJ','Tvn[K]': 'Tvn'})

dfs['LR']=df['LR']

dfs_train=splitmixture(dfs,a,n)[0]
dfs_test=splitmixture(dfs,a,n)[1]

dfs_train = dfs_train.sample(frac=1).reset_index(drop=True)

X_train = dfs_train.drop('LR', axis=1)
y_train =  dfs_train['LR']

X_test = dfs_test.drop('LR', axis=1)
y_test = dfs_test['LR']


sol=['adam']
act=['relu']
hidd=[]

for i in [4,6,8]:
    for j in [100,150,200]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)
alp=[1e-1]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
grid=GridSearchCV(MLPRegressor(),param_grid,cv=3)
grid.fit(X_train,y_train)
result=grid.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
print('Test set score: {}'.format(grid.score(X_test, y_test)))
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))



result = permutation_importance(grid, X_test, y_test, n_repeats=5, random_state=42)


df_importance = pd.DataFrame(zip(X_train.columns, result["importances"].mean(axis=1)),columns=["feature","importance"])
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
df_importance.to_excel('/mnt/c/CEA/df_importance1.xlsx')


plt.figure(figsize=(10,10))

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
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig("gurafu5.png")






## 11parameters
#######################################################

df = pd.read_excel('all_mixture.xlsx')

output=df['LR']

l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Tvn[K]']

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params


#dfs=dfs.rename(columns={'p0[bar]': 'p0', 'H0[KJ/kg]': 'H0','M0kg/kmol': 'M0','gamma_0': 'γ0','p_CJ[bar]': 'pCJ', 'T_CJ[K]': 'TCJ', 'H_CJ[KJ/kg]': 'HCJ',  'M_CJkg/kmol': 'MCJ', 'gamma_CJ': 'γCJ', 'M_CJ': 'M_CJ','Tvn[K]': 'Tvn'})

dfs['LR']=df['LR']

dfs_train=splitmixture(dfs,a,n)[0]
dfs_test=splitmixture(dfs,a,n)[1]

dfs_train = dfs_train.sample(frac=1).reset_index(drop=True)

X_train = dfs_train.drop('LR', axis=1)
y_train =  dfs_train['LR']

X_test = dfs_test.drop('LR', axis=1)
y_test = dfs_test['LR']

sol=['adam']
act=['relu']
hidd=[]

for i in [4,6,8]:
    for j in [100,150,200]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)
alp=[1e-1]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
grid=GridSearchCV(MLPRegressor(),param_grid,cv=10)
grid.fit(X_train,y_train)
result=grid.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
print('Test set score: {}'.format(grid.score(X_test, y_test)))
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))


result = permutation_importance(grid, X_test, y_test, n_repeats=5, random_state=42)


df_importance = pd.DataFrame(zip(X_train.columns, result["importances"].mean(axis=1)),columns=["feature","importance"])
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
df_importance.to_excel('/mnt/c/CEA/df_importance2.xlsx')


plt.figure(figsize=(8,5))

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
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig("gurafu6.png")


















'''
##ランダムフォレストで予測モデルの構築
#######################################################
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=440)

clf = RandomForestRegressor(n_estimators=30,max_depth=9)
#print("Cross Val Score (MSE):"+str(cross_val_score(clf, data_X, data_y, cv=5, scoring='r').mean()))

clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(mean_squared_error(y_test, pred))

## Gini Importanceの可視化
df_importance = pd.DataFrame(zip(X_train.columns, clf.feature_importances_),columns=["Features","Importance"])
df_importance = df_importance.sort_values("Importance",ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x="Importance", y="Features",data=df_importance,ci=None)
plt.title("Gini Importance")
plt.tight_layout()
plt.savefig("gurafu5.png")

result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
result

df_importance = pd.DataFrame(zip(X_train.columns, result["importances"].mean(axis=1)),columns=["Features","Importance"])
df_importance = df_importance.sort_values("Importance",ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x="Importance", y="Features",data=df_importance,ci=None)
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig("gurafu6.png")

'''