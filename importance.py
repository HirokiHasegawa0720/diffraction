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
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
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


rf1 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

rf1.fit(X_train, y_train)

# Feature Importances
fti =  rf1.feature_importances_

print('Feature Importances:')

cols = list(X .columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(fti) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df1=df_importance
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
df_importance.to_excel('/mnt/c/CEA/Gini importance(rondomforest).xlsx')

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
plt.savefig("gurafu3.png")




#lightgbm

gbm_reg1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31,
                        n_estimators=100)
gbm_reg1.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l2',
        verbose=0)


# Feature Importances
fti = gbm_reg1.feature_importances_

cols = list(X.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(fti) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df2=df_importance
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
df_importance.to_excel('/mnt/c/CEA/Gini importance(lightgbm).xlsx')

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
plt.title("Gini importance(lightgbm)")
plt.tight_layout()
plt.savefig("gurafu4.png")







rf2 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

rf2.fit(X_train, y_train)

result = permutation_importance(rf2,X_train, y_train, n_repeats=5, random_state=42)

cols = list(X.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(result["importances"].mean(axis=1)) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df3=df_importance
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
df_importance.to_excel('/mnt/c/CEA/Permutation Importance(rondomforst).xlsx')


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
plt.title("Permutation Importance(rondomforst)")
plt.tight_layout()
plt.savefig("gurafu5.png")









gbm_reg2 = lgb.LGBMRegressor(objective='regression',
                        num_leaves = 31,
                        n_estimators=100)
gbm_reg2.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l2',
        verbose=0)


result = permutation_importance(gbm_reg2,X_train, y_train, n_repeats=5, random_state=42)


cols = list(X.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(result["importances"].mean(axis=1)) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df4=df_importance
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
df_importance.to_excel('/mnt/c/CEA/Permutation Importance(lightgbm).xlsx')


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
plt.title("Permutation Importance(lightgbm)")
plt.tight_layout()
plt.savefig("gurafu6.png")











sol=['adam']
act=['relu']
hidd=[]

for i in [4]:
    for j in [100]:
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




result = permutation_importance(grid,X_train,y_train, n_repeats=5, random_state=42)


cols = list(X.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(result["importances"].mean(axis=1)) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df5=df_importance
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
df_importance.to_excel('/mnt/c/CEA/Permutation Importance(mlpregressor).xlsx')


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
plt.title("Permutation Importance(mlpregressor)")
plt.tight_layout()
plt.savefig("gurafu7.png")

'''
df1 = pd.read_excel('/mnt/c/CEA/Gini importance(rondomforest).xlsx')
df2 = pd.read_excel('/mnt/c/CEA/Gini importance(lightgbm).xlsx')
df3 = pd.read_excel('/mnt/c/CEA/Permutation Importance(rondomforst).xlsx')
df4 = pd.read_excel('/mnt/c/CEA/Permutation Importance(lightgbm).xlsx')
df5 = pd.read_excel('/mnt/c/CEA/Permutation Importance(mlpregressor).xlsx')
'''




f_importance=[(x + y)/2 for (x, y) in zip(df1['importance'], df2['importance'])]

cols = list(X.columns) 


df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})

print(df_importance)
df_importance.to_excel('/mnt/c/CEA/Permutation Importance(mlpregressor).xlsx')


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
plt.title("Gini importance(Rondomforest and lightgbm).xlsx")
plt.tight_layout()
plt.savefig("gurafu8.png")






f_importance=[(x + y + z)/3 for (x, y, z) in zip(df3['importance'], df4['importance'], df5['importance'])]

cols = list(X.columns) 

df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})

print(df_importance)

df_importance.to_excel('/mnt/c/CEA/Permutation importance(Rondomforest and lightgbm and mlp regressor).xlsx')


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
plt.title("Permutation importance(Rondomforest and lightgbm and mlp regressor).xlsx")
plt.tight_layout()
plt.savefig("gurafu9.png")





f_importance=[(x + y + z + k + l )/5 for (x, y, z ,k ,l) in zip(df1['importance'], df2['importance'],df3['importance'], df4['importance'], df5['importance'])]

cols = list(X.columns) 

df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})

print(df_importance)

df_importance.to_excel('/mnt/c/CEA/Permutation and Gini.xlsx')


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
plt.title("Permutation and Gini.xlsx")
plt.tight_layout()
plt.savefig("gurafu10.png")





