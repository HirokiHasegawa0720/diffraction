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

df = pd.read_excel('/mnt/c/CEA/all_mixture.xlsx')

#3.1　特徴量の削減

l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'a0[m/s]', \
        'pcj[bar]', 'Tcj[K]', 'Hcj[KJ/kg]',\
                    'Mcj[kg/kmol]', 'γcj[-]', 'acj[m/s]',\
                         'Mcj[-]', 'Vcj[m/s]',\
        'Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', \
                      'γvn[-]','avn[m/s]',\
                    'VISCMILLIPOISEvn','CONDUCTIVITYvn','PRANDTLNUMBERvn','VISCMILLIPOISECJ','CONDUCTIVITYCJ','PRANDTLNUMBERCJ']

dfs = pd.DataFrame({'p0[bar]':df[l[0]]})

for i in range(len(l)):
    params=df[l[i]]
    dfs[l[i]]=params

x = dfs
y = df['LR']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

dfss=dfs
dfss['reflection']=df['LR']

threshold = 0.8

feat_corr = set()
corr_matrix = X_train.corr()
corr_matrix.to_excel('/mnt/c/CEA/matrix1.xlsx')
plt.figure()
sns.heatmap(corr_matrix,cmap='viridis')
plt.tight_layout()
plt.savefig('gurafu1(研究報告).png')
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


def seikika1(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))

def seikika2(pop):
    s_pop = (pop - pop.mean()) / pop.std()
    return s_pop 

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
a=20  #C2H4/O2,AR50

def splitmixture1(dfs,a,n):
        train = dfs.drop(range(a*n,(a+1)*n))
        test = dfs[a*n:(a+1)*n]
        return train,test

df = pd.read_excel('/mnt/c/CEA/all_mixture.xlsx')

l=[ 'p0[bar]','Tcj[K]']

'''
l=['p0[bar]', 'H0[KJ/kg]', 'M0[kg/kmol]', 'γ0[-]', 'pcj[bar]', 'Tcj[K]',
       'γcj[-]', 'Mcj[-]', 'VISCMILLIPOISECJ', 'PRANDTLNUMBERCJ']
'''

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika2(df[l[i]])
    dfs[l[i]]=params


dfs['LR']=df['LR']

dfs_train=splitmixture1(dfs,a,n)[0]
dfs_test=splitmixture1(dfs,a,n)[1]

X_val = dfs_train.drop('LR', axis=1)
y_val =  dfs_train['LR']

X_test2 = dfs_test.drop('LR', axis=1)
y_test2 = dfs_test['LR']

X_train, X_test1, y_train, y_test1 = train_test_split(X_val , y_val, train_size=0.6, random_state=123)
'''

print(X_train)

X_train = X_train.sort_index()
y_train = y_train.sort_index()


print(X_train)

'''

mms = MinMaxScaler()
ss = StandardScaler()

sX_train = ss.fit_transform(X_train)
 
sX_test1 = ss.transform(X_test1)

sX_test2 = ss.fit_transform(X_test2)



#MSE vs エポック

sol=['adam']
act=['relu']
hidd=[(200,200,200,200)]
#bach=[30]

for i in [4,6]:
    for j in [100,150,200]:
        b=[j]*i
        b=tuple(b)
        hidd.append(b)


alp=[1e-4,1e+0]
#param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp,'batch_size':bach}
param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
grid=GridSearchCV(MLPRegressor(), param_grid , cv=2, n_jobs=4)
grid.fit(X_train,y_train)
result1mlp=grid.predict(X_test1)
MSE1=mean_squared_error(y_test1, result1mlp)
R21=r2_score(y_test1,result1mlp)
print('Test set score: {}'.format(grid.score(X_test1, y_test1)))
print('Best parameters: {}'.format(grid.best_params_))
print('Best cross-validation: {}'.format(grid.best_score_))
result2mlp=grid.predict(X_test2)
MSE2=mean_squared_error(y_test2, result2mlp)
R22=r2_score(y_test2,result2mlp)
print(y_test2)
print(MSE1,R21)
print(MSE2,R22)

plt.figure()
cv_result = pd.DataFrame(grid.cv_results_)
cv_result = cv_result[['param_hidden_layer_sizes', 'param_alpha', 'mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_hidden_layer_sizes', 'param_alpha')
heat_map = sns.heatmap(cv_result_pivot, cmap='viridis', annot=True)
plt.tight_layout()
plt.savefig("gurafu2(研究報告).png")
plt.close('all')
''''

param_grid = { "n_estimators":[100,500,1000,1500],'max_depth':[10,20,30,40]}
forest_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                 param_grid = param_grid,   
                 scoring="r2",  #metrics
                 cv = 2,              #cross-validation
                 n_jobs = 4)          #number of core

forest_grid.fit(X_train,y_train)
result1rondom=forest_grid.predict(X_test1)
MSE1=mean_squared_error(y_test1, result1rondom)
R21=r2_score(y_test1,result1rondom)
print('Test set score: {}'.format(forest_grid.score(X_test1, y_test1)))
print('Best parameters: {}'.format(forest_grid.best_params_))
print('Best cross-validation: {}'.format(forest_grid.best_score_))
result2rondom=forest_grid.predict(X_test2)
MSE2=mean_squared_error(y_test2, result2rondom)
R22=r2_score(y_test2,result2rondom)
print(MSE1,R21)
print(MSE2,R22)

plt.figure()
cv_result = pd.DataFrame(forest_grid.cv_results_)
cv_result = cv_result[['param_n_estimators', 'param_max_depth', 'mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_n_estimators', 'param_max_depth')
cv_result_pivot.to_excel('/mnt/c/CEA/cv_result_pivot.xlsx')
heat_map = sns.heatmap(cv_result_pivot, cmap='viridis', annot=True)
plt.tight_layout()
plt.savefig("gurafu3(研究報告).png")
plt.close('all')



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

values =  np.concatenate([y_test1, result1mlp], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1


plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_test1,result1mlp,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment[mm]')
plt.ylabel('LR predict[mm]')
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


values =  np.concatenate([y_test2, result2mlp], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_test2,result2mlp,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment[mm]')
plt.ylabel('LR predict[mm]')
plt.title("mlp")
plt.tight_layout()

plt.savefig("gurafu5(研究報告).png")


#rondom

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

values =  np.concatenate([y_test1, result1rondom], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1


plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_test1,result1rondom,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment[mm]')
plt.ylabel('LR predict[mm]')
plt.title("rondomforest")
plt.tight_layout()
plt.savefig("gurafu6(研究報告).png")


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


values =  np.concatenate([y_test2, result2rondom], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

plt.plot([min_value, max_value], [min_value, max_value],color='black')

plt.scatter(y_test2,result2rondom,s=30)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('LR experiment[mm]')
plt.ylabel('LR predict[mm]')
plt.title("rondomforest")
plt.tight_layout()
plt.savefig("gurafu7(研究報告).png")




#3.4特徴量の寄与度
print(grid.best_estimator_.hidden_layer_sizes)
estimator1 = MLPRegressor(hidden_layer_sizes=grid.best_estimator_.hidden_layer_sizes,alpha=grid.best_estimator_.alpha,solver='adam',activation='relu')

estimator1.fit(X_train,y_train)

#順列重要度vs特徴量


result = permutation_importance(grid,X_train,y_train, n_repeats=5, random_state=42)

cols = list(X_train.columns)         # 特徴量名のリスト(目的変数CRIM以外)
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
plt.savefig("gurafu8(研究報告).png")



estimator2 =RandomForestRegressor(n_estimators=forest_grid.best_estimator_.n_estimators,max_depth=forest_grid.best_estimator_.max_depth,
                                random_state=42)
estimator2.fit(X_train,y_train)
fti =  estimator2.feature_importances_

print('Feature Importances:')

cols = list(X_train .columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(fti) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df2=df_importance
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート


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


plot_feature_importance(df_importance)
plt.title("Gini importance(rondomforest)")
plt.tight_layout()
plt.savefig("gurafu9(研究報告).png")

#Gini and permutation importance



f_importance=[(x + y)/2 for (x, y) in zip(df1['importance'], df2['importance'])]

cols = list(X_train.columns) 


df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})

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
plt.title("Gini importance and permutation importance.xlsx")
plt.tight_layout()
plt.savefig("gurafu10(研究報告).png")



from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt


# 確認したい特徴量を記述
features = ['p0[bar]', 'Tcj[K]', 'γcj[-]']

# plot                        
#plot_partial_dependence(estimator=estimator, X=X_train, features=features,
#                        n_jobs=-4, grid_resolution=20, n_cols=2)

plt.figure(figsize=(8,8))

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



tree_disp = plot_partial_dependence(estimator1, X_train, ['p0[bar]'],line_kw={"label": "Neural Network"})

mlp_disp = plot_partial_dependence(estimator2, X_train, ['p0[bar]'],
                                   ax=tree_disp.axes_,
                                   line_kw={"label": "Random forest","color": "red"})

plt.tight_layout()
plt.savefig("gurafu11(研究報告).png")


plt.figure(figsize=(8,8))


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


tree_disp = plot_partial_dependence(estimator1, X_train, ['Tcj[K]'],line_kw={"label": "Neural Network"})
mlp_disp = plot_partial_dependence(estimator2, X_train, ['Tcj[K]'],
                                   ax=tree_disp.axes_,
                                   line_kw={"label": "Random forest","color": "red"})

plt.tight_layout()
plt.savefig("gurafu12(研究報告).png")



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

tree_disp = plot_partial_dependence(estimator1, X_train, ['γcj[-]'],line_kw={"label": "neuralnetwork"})
mlp_disp = plot_partial_dependence(estimator2, X_train, ['γcj[-]'],
                                   ax=tree_disp.axes_,
                                   line_kw={"color": "red"})

plt.tight_layout()
plt.savefig("gurafu13(研究報告).png")


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

tree_disp = plot_partial_dependence(estimator1, X_train, ['M0[kg/kmol]'])
mlp_disp = plot_partial_dependence(estimator2, X_train, ['M0[kg/kmol]'],
                                   ax=tree_disp.axes_,
                                   line_kw={"color": "red"})

plt.tight_layout()
plt.savefig("gurafu14(研究報告).png")


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

tree_disp = plot_partial_dependence(estimator1, X_train, ['Mcj[-]'])
mlp_disp = plot_partial_dependence(estimator2, X_train, ['Mcj[-]'],
                                   ax=tree_disp.axes_,
                                   line_kw={"color": "red"})

plt.tight_layout()
plt.savefig("gurafu15(研究報告).png")

plt.figure(figsize=(5,5))
plt.hist(dfs['p0[bar]'])
plt.title('p0[bar]')
plt.tight_layout()
plt.savefig("gurafu16(研究報告).png")

plt.figure(figsize=(5,5))
plt.title('Tcj[K]')
plt.hist(dfs['Tcj[K]'])
plt.tight_layout()
plt.savefig("gurafu17(研究報告).png")

plt.figure(figsize=(5,5))
plt.title('γcj[-]')
plt.hist(dfs['γcj[-]'])
plt.tight_layout()
plt.savefig("gurafu18(研究報告).png")

plt.figure(figsize=(5,5))
plt.title('M0[kg/kmol]')
plt.hist(dfs['M0[kg/kmol]'])
plt.tight_layout()
plt.savefig("gurafu19(研究報告).png")

plt.figure(figsize=(5,5))
plt.title('Mcj[-]')
plt.hist(dfs['Mcj[-]'])
plt.tight_layout()
plt.savefig("gurafu20(研究報告).png")

plt.figure(figsize=(5,5))
plt.title('γ0[-]')
plt.hist(dfs['γ0[-]'])
plt.tight_layout()
plt.savefig("gurafu21(研究報告).png")

'''
