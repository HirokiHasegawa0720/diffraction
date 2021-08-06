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
from sklearn.inspection import permutation_importance


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

#dfs=dfs.rename(columns={'p0[bar]': 'p0', 'H0[KJ/kg]': 'H0','M0kg/kmol': 'M0','gamma_0': 'γ0','p_CJ[bar]': 'pCJ', 'T_CJ[K]': 'TCJ', 'H_CJ[KJ/kg]': 'HCJ',  'M_CJkg/kmol': 'MCJ', 'gamma_CJ': 'γCJ', 'M_CJ': 'M_CJ','Tvn[K]': 'Tvn'})

dfs['LR']=df['LR']

dfs_train=splitmixture(dfs,a,n)[0]
dfs_test=splitmixture(dfs,a,n)[1]

dfs_train = dfs_train.sample(frac=1).reset_index(drop=True)

X_train = dfs_train.drop('LR', axis=1)
y_train =  dfs_train['LR']

X_test = dfs_test.drop('LR', axis=1)
y_test = dfs_test['LR']

#Rondomforest


rf1 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

rf1.fit(X_train, y_train)

# Feature Importances
fti =  rf1.feature_importances_

print('Feature Importances:')
for i, feat in enumerate(X_train .columns):
    print('\t{0:10s} : {1:>12.4f}'.format(feat, fti[i]))

cols = list(X_train .columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(fti) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート

display(df_importance)
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


plot_feature_importance(df_importance)
plt.title("Gini importance")
plt.tight_layout()
plt.savefig("gurafu7.png")







rf2 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

rf2.fit(X_train, y_train)

result = permutation_importance(rf2,X_train, y_train, n_repeats=5, random_state=42)


df_importance = pd.DataFrame(zip(X_train.columns, result["importances"].mean(axis=1)),columns=["feature","importance"])
df_importance = df_importance.sort_values("importance",ascending=False)
print(df_importance)
#df_importance.to_excel('/mnt/c/CEA/df_importance1.xlsx')


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
plt.savefig("gurafu8.png")

