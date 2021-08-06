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

df = pd.read_excel('all_mixture.xlsx')


#3.1　特徴量の削減

l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'a0[m/s]', \
        'pcj[bar]', 'Tcj[K]', 'Hcj[KJ/kg]',\
                    'Mcj[kg/kmol]', 'γcj[-]', 'acj[m/s]',\
                         'Mcj[-]', 'Vcj[m/s]',\
        'Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', \
                      'γvn[-]','avn[m/s]', \
                        ]


dfs = pd.DataFrame({'p0[bar]':df[l[0]]})

for i in range(len(l)):
    params=df[l[i]]
    dfs[l[i]]=params

x = dfs
y = df['LR']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=123)

dfss=dfs
dfss['reflection']=df['LR']

threshold = 0.8

feat_corr = set()
corr_matrix = X_train.corr()
corr_matrix.to_excel('matrix1.xlsx')
plt.figure()
sns.heatmap(corr_matrix,cmap='Greys')
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
        

X_train.drop(labels=feat_corr, axis='columns', inplace=True)
X_test.drop(labels=feat_corr, axis='columns', inplace=True)
print(X_train.columns)
print(len(X_train.columns))

corr_matrix.to_excel('matrix2.xlsx')







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

df = pd.read_excel('all_mixture.xlsx')


l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'Tcj[K]', 'γcj[-]',
       'Mcj[-]']

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params



dfs['LR']=df['LR']

dfs_train=splitmixture(dfs,a,n)[0]
dfs_test=splitmixture(dfs,a,n)[1]

X_val = dfs_train.drop('LR', axis=1)
y_val =  dfs_train['LR']

X_test2 = dfs_test.drop('LR', axis=1)
y_test2 = dfs_test['LR']


X_train, X_test1, y_train, y_test1 = train_test_split(X_val , y_val, train_size=0.6, random_state=123)



#MSE vs エポック
'''

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=2, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 10)):
    print(train_sizes)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Number of training samples', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print(train_sizes)
    
    return train_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std
'''

sol=['adam']
act=['relu']
hidd=[(200,200,200,200)]
#bach=[30]

for i in [4,6,8]:
    for j in [100,150,200]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)


alp=[1e-4,1e-1]
#param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp,'batch_size':bach}
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
grid=GridSearchCV(MLPRegressor(),param_grid,cv=2)
grid.fit(X_train,y_train)
result1=grid.predict(X_test1)
MSE1=mean_squared_error(y_test1, result1)
R21=r2_score(y_test1,result1)
print(MSE1,R21)
print('Test set score: {}'.format(grid.score(X_test1, y_test1)))
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))
result2=grid.predict(X_test2)
MSE2=mean_squared_error(y_test2, result2)
R22=r2_score(y_test2,result2)
print(MSE2,R22)

plt.figure()
cv_result = pd.DataFrame(grid.cv_results_)
cv_result.to_excel('/mnt/c/CEA/gurafu1.xlsx')
cv_result = cv_result[['param_hidden_layer_sizes', 'param_alpha', 'mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_hidden_layer_sizes', 'param_alpha')
print(cv_result_pivot )
heat_map = sns.heatmap(cv_result_pivot, cmap='Greys', annot=True)
plt.tight_layout()
plt.savefig("gurafu1.png")
plt.close('all')

title = 'Learning Curves (MLPRegressor)' 
estimator = MLPRegressor(hidden_layer_sizes=grid.best_estimator_.hidden_layer_sizes,alpha=grid.best_estimator_.alpha,solver='adam',activation='relu')




'''
plot_learning_curve(estimator, title, X_train, y_train)
train_sizes=plot_learning_curve(estimator, title, X_train, y_train)[0]
train_scores_mean=plot_learning_curve(estimator, title, X_train, y_train)[1]
train_scores_std=plot_learning_curve(estimator, title, X_train, y_train)[2]
test_scores_mean=plot_learning_curve(estimator, title, X_train, y_train)[3]
test_scores_std=plot_learning_curve(estimator, title, X_train, y_train)[4]

plt.figure()

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

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")


plt.savefig("gurafu2.png")


print(len( X_train))
'''



#3.3 あてはめ性能
#Ypredict vs Yexp(学習データ)


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

values =  np.concatenate([y_test1, result1], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1


plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_test1,result1,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment')
plt.ylabel('LR predict')
plt.tight_layout()

plt.savefig("gurafu3.png")


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


values =  np.concatenate([y_test2, result2], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_test2,result2,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment')
plt.ylabel('LR predict')
plt.tight_layout()

plt.savefig("gurafu4.png")








#3.4特徴量の寄与度

#順列重要度vs特徴量



result = permutation_importance(grid,X_train,y_train, n_repeats=5, random_state=42)


df_importance = pd.DataFrame(zip(X_train.columns, result["importances"].mean(axis=1)),columns=["feature","importance"])
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
df_importance.to_excel('/mnt/c/CEA/df_importance1.xlsx')


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
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig("gurafu5.png")


#Yexp vs  Tcj


n=10  #1つの混合気あたりのデータ点数
 #H2/O2

def splitmixture(dfs,a,n):
        train = dfs.drop(range(a*n,(a+1)*n))
        test = dfs[a*n:(a+1)*n]
        return train,test

df = pd.read_excel('all_mixture.xlsx')


l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'Tcj[K]', 'γcj[-]',
       'Mcj[-]']

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params



dfs['LR']=df['LR']

a=4#H2O2
dfH2O2=splitmixture(dfs,a,n)[1]
H2O2Tcj = dfH2O2['Tcj[K]']
H2O2LR = dfH2O2['LR']

a=34
dfC2H4AR50=splitmixture(dfs,a,n)[1]
C2H4AR50Tcj = dfC2H4AR50['Tcj[K]']
C2H4AR50LR = dfC2H4AR50['LR']

a=37
dfC2H2Kr=splitmixture(dfs,a,n)[1]
C2H2KrTcj = dfC2H2Kr['Tcj[K]']
C2H2KrLR = dfC2H2Kr['LR']



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
plt.scatter(H2O2Tcj, H2O2LR ,marker='o',s=60,color='red')
plt.plot(H2O2Tcj,H2O2LR,color='red',linestyle='--',marker='')
plt.scatter(C2H4AR50Tcj , C2H4AR50LR ,marker='o',s=60,color='blue')
plt.plot(C2H4AR50Tcj,C2H4AR50LR,color='blue',linestyle='--',marker='')
plt.scatter(C2H2KrTcj, C2H2KrLR ,marker='o',s=60,color='black')
plt.plot(C2H2KrTcj,C2H2KrLR,color='black',linestyle='--',marker='')
plt.xlabel('LR experiment')
plt.ylabel('Tcj[k]')
plt.tight_layout()
plt.savefig("gurafu6.png")

