from sklearn.feature_selection import RFE
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
import lightgbm as lgb #LightGBM
# データフレームを綺麗に出力する関数
import IPython
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV



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

df = pd.read_excel('/mnt/c/CEA/all_mixture.xlsx')

output=df['LR']

l=['p0[bar]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0','p_CJ[bar]', 'T_CJ[K]', 'H_CJ[KJ/kg]', 'M_CJkg/kmol', 'gamma_CJ','M_CJ', 'Tvn[K]']

dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params



dfs=dfs.rename(columns={'p0[bar]': 'p0', 'H0[KJ/kg]': 'H0','M0kg/kmol': 'M0','gamma_0': 'γ0','p_CJ[bar]': 'pCJ', 'T_CJ[K]': 'TCJ', 'H_CJ[KJ/kg]': 'HCJ',  'M_CJkg/kmol': 'MCJ', 'gamma_CJ': 'γCJ', 'M_CJ': 'M_CJ','Tvn[K]': 'Tvn'})

X = dfs
y = output


X_train , X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=123)

print(dfs)


#lightgbm

gbm_reg1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31,
                        n_estimators=100)
in_features_to_select = 1
selector = RFECV(gbm_reg1 , min_features_to_select=in_features_to_select, cv=3,scoring='r2')
selector = selector.fit(X_train ,  y_train)
X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])
print(X_new)
result = pd.DataFrame(selector.get_support(), index=X.columns.values, columns=['False: dropped'])
print(result)
result['ranking'] = selector.ranking_
print(result)

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
plt.savefig("gurafu3.png")


Xnew_train , Xnew_test, ynew_train, ynew_test = train_test_split(X_new, y, train_size=0.6, random_state=123)


param_grid ={'n_estimators':[100,200,300,400,500],'max_depth':[10,100,50],'min_data_in_leaf':[100,300,500], 'num_leaves':[10,100,30],'learning_rate':[0.1,0.05,0.01]}

forest_grid = GridSearchCV(lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31),
                 param_grid = param_grid,   
                 scoring="r2",  #metrics
                 cv = 3,              #cross-validation
                 n_jobs = 1)          #number of core

forest_grid.fit(X_train,y_train) #fit
result = forest_grid.predict(X_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)



forest_grid.fit(Xnew_train, ynew_train) #fit
result = forest_grid.predict(Xnew_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)


'''







gbm_reg2 = lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31,
                        n_estimators=100)
gbm_reg2.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l2',
        verbose=0)

# Feature Importances
fti = gbm_reg2.feature_importances_

print('Feature Importances:')
for i, feat in enumerate(X.columns):
    print('\t{0:10s} : {1:>12.4f}'.format(feat, fti[i]))

cols = list(X.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(fti) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)

plt.figure(figsize=(6,5))




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
plt.savefig("gurafu4.png")


'''