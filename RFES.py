from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
# ライブラリのインポート
import matplotlib.pyplot as plt # グラフ描画用
#import seaborn as sns; sns.set() # グラフ描画用
import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')
# データフレームを綺麗に出力する関数
import IPython
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb 


def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))


def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)


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


l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Tvn[K]']

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params

dff=dfs.rename(columns={'p0[bar]': 'p0', 'H0[KJ/kg]': 'H0','M0[kg/kmol]': 'M0','γ0[-]': 'γ0','pcj[bar]': 'pCJ', 'Tcj[K]': 'TCJ', 'Hcj[KJ/kg]': 'HCJ',  'Mcj[kg/kmol]': 'MCJ', 'γcj[-]': 'γCJ', 'Mcj[-]': 'M_CJ','Tvn[K]': 'Tvn'})

dff['LR']=df['LR']

dff_train=splitmixture(dff,a,n)[0]
dff_test=splitmixture(dff,a,n)[1]

dff_train = dff_train.sample(frac=1).reset_index(drop=True)

X_train = dff_train.drop('LR', axis=1)
y_train =  dff_train['LR']

X_test = dff_test.drop('LR', axis=1)
y_test = dff_test['LR']

X=dfs
y=df['LR']


#Rondomforest



rf2 = RandomForestRegressor(n_estimators=500,max_depth=7,
                                random_state=42)

in_features_to_select = 1
selector = RFECV(rf2, min_features_to_select=in_features_to_select, cv=3,scoring='r2')
selector = selector.fit(X_train, y_train)

plt.figure(figsize=(6,5))

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 4.0

plt.rcParams['axes.linewidth'] = 2.0

# Tick Setting
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
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (r2)")
plt.title("Rondomforest")
plt.ylim(0,1)
plt.plot(range(in_features_to_select,
               len(selector.grid_scores_) + in_features_to_select),
         selector.grid_scores_)
plt.savefig("gurafu1.png")

print(selector.grid_scores_)


X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X_train.columns.values[selector.get_support()])

print(X_train.columns.values[selector.get_support()])

result = pd.DataFrame(selector.get_support(), index=X_train.columns.values, columns=['False: dropped'])
result['ranking'] = selector.ranking_


#allpameter


d = {'model(parameter)':['1'] , 'MSE':[0],'R2':[0]}
dfsss = pd.DataFrame(d,columns=['model(parameter)','MSE','R2'])
#7

print('romdomforest')
param_grid = { "n_estimators":[100,500,1000],'max_depth':[3,5,7,9]}


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

dfsss.loc[0]=['Randomforest(All)',MSE,R2]

print('mlp')
sol=['adam']
act=['relu']
hidd=[]
for i in [2,4,6]:
    for j in [50,100,150]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)
alp=[1-4,1e-2,1e+0]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
grid=GridSearchCV(MLPRegressor(),param_grid,cv=3)
grid.fit(X_train,y_train)
result=grid.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))

dfsss.loc[1]=['MLPRegressor(All)',MSE,R2]


selector = RFE(rf2, n_features_to_select=6)
selector = selector.fit(X,  y)
X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])
print(X.columns.values[selector.get_support()])

dfss=X_new

dfss['LR']=y


dfss_train=splitmixture(dfss,a,n)[0]
dfss_test=splitmixture(dfss,a,n)[1]

dfss_train = dfss_train.sample(frac=1).reset_index(drop=True)

Xnew_train = dfss_train.drop('LR', axis=1)
ynew_train =  dfss_train['LR']

Xnew_test = dfss_test.drop('LR', axis=1)
ynew_test = dfss_test['LR']

print('romdomforest')

forest_grid.fit(Xnew_train, ynew_train) #fit
result = forest_grid.predict(Xnew_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))

dfsss.loc[2]=['Romdomforest(RFE(Romdomforest))',MSE,R2]


print('mlp')

grid.fit(Xnew_train, ynew_train)
result=grid.predict(Xnew_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))

dfsss.loc[3]=['MLPRegressor(RFE(Romdomforest))',MSE,R2]



## allparameters
#######################################################



#l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
#       'Hcj[KJ/kg]', 'Mcj[kg/kmol]', 'γcj[-]', 'Mcj[-]', 'Tvn[K]']


#lightgbm



gbm_reg2 = lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31,
                        n_estimators=100)

in_features_to_select = 1
selector = RFECV(gbm_reg2 , min_features_to_select=in_features_to_select, cv=3,scoring='r2')
selector = selector.fit(X_train, y_train)

print(selector.grid_scores_)

plt.figure(figsize=(6,5))

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 4.0

plt.rcParams['axes.linewidth'] = 2.0

# Tick Setting
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
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (r2)")
plt.plot(range(in_features_to_select,
               len(selector.grid_scores_) + in_features_to_select),
         selector.grid_scores_)
plt.ylim(0,1)
plt.title("Lightgbm")
plt.savefig("gurafu2.png")
print(selector.grid_scores_)

X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])

print(X_new.columns)


#all parameters

#7

print('lightgbm')

#param_grid ={'n_estimators':[600,800,1000],'max_depth':[2,4,6],'min_data_in_leaf':[5,10,15], 'num_leaves':[4,6,8],'learning_rate':[0.1,0.2,0.3]}
param_grid ={'n_estimators':[100,500,100],'max_depth':[2,4,6]}

forest_grid = GridSearchCV(lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31),
                 param_grid = param_grid,   
                 scoring="r2",  #metrics
                 cv = 3)    


forest_grid.fit(X_train,y_train) #fit
result = forest_grid.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))

dfsss.loc[4]=['lightgbm(All)',MSE,R2]

print('mlp')

sol=['adam']
act=['relu']
hidd=[]
for i in [2,4,6]:
    for j in [50,100,150]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)
alp=[1-4,1e-2,1e+0]
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
grid=GridSearchCV(MLPRegressor(),param_grid,cv=3)
grid.fit(X_train,y_train)
result=grid.predict(X_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))

dfsss.loc[5]=['MLPRegressor(All)',MSE,R2]


selector = RFE(gbm_reg2, n_features_to_select=6)
selector = selector.fit(X,  y)
X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])
print(X.columns.values[selector.get_support()])

dfss=X_new

dfss['LR']=y


dfss_train=splitmixture(dfss,a,n)[0]
dfss_test=splitmixture(dfss,a,n)[1]

dfss_train = dfss_train.sample(frac=1).reset_index(drop=True)

Xnew_train = dfss_train.drop('LR', axis=1)
ynew_train =  dfss_train['LR']

Xnew_test = dfss_test.drop('LR', axis=1)
ynew_test = dfss_test['LR']

print('lightgbm')

forest_grid.fit(Xnew_train, ynew_train) #fit
result = forest_grid.predict(Xnew_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))

dfsss.loc[6]=['lightgbm(RFE(lightgbm))',MSE,R2]


print('mlp')

grid.fit(Xnew_train, ynew_train)
result=grid.predict(Xnew_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))

dfsss.loc[7]=['MLPRegressor(RFE(lightgbm))',MSE,R2]

dfsss.to_excel('/mnt/c/CEA/predictions.xlsx')