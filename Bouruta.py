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
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy





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
'''
l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Tvn[K]']
'''

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




param_grid = { "n_estimators":[100,500,1000],'max_depth':[10,50,100]}


forest_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                 param_grid = param_grid,   
                 scoring="r2",  #metrics
                 cv = 3,              #cross-validation
                 n_jobs = 1)          #number of core

forest_grid.fit(X_train,y_train) #fit
result = forest_grid.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))


'''
rf1 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

rf1.fit(X_train, y_train)

result=rf1.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)

'''





rf2= RandomForestRegressor(n_estimators=1000,
                                random_state=42)

feat_selector = BorutaPy(rf2, n_estimators='auto', two_step=False,verbose=2, random_state=42)
# two_stepがない方、つまりBonferroniを用いたほうがうまくいく

print(X_train.values)
# データの二度漬けになるので特徴量選択する際にもtestを含めてはいけない
feat_selector.fit(X_train.values,y_train.values)

print(2)
X_train_selected = X_train.iloc[:,feat_selector.support_]

X_test_selected = X_test.iloc[:,feat_selector.support_]

print(X_train_selected.columns)

print(3)




param_grid = { "n_estimators":[100,500,1000],'max_depth':[10,50,100]}


forest_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                 param_grid = param_grid,   
                 scoring="r2",  #metrics
                 cv = 3,#cross-validation
                 n_jobs = 1)#number of core
                 

forest_grid.fit(X_train_selected,y_train) #fit
result = forest_grid.predict(X_test_selected)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))


'''
rf=RandomForestRegressor(
    n_estimators=500,
    random_state=42,
)
rf.fit(X_train_selected, y_train)

print(X_train_selected.values)

result=rf.predict(X_test_selected)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)

'''