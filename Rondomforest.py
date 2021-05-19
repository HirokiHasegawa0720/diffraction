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


#Rondomforest


rf1 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

rf1.fit(X_train, y_train)

# Feature Importances
fti =  rf1.feature_importances_

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
plt.savefig("gurafu1.png")









rf2 = RandomForestRegressor(n_estimators=100,
                                random_state=42)

in_features_to_select = 1
selector = RFECV(rf2, min_features_to_select=in_features_to_select, cv=5,scoring='r2')
selector = selector.fit(X,  y)
X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])

print(X_new)
result = pd.DataFrame(selector.get_support(), index=X.columns.values, columns=['False: dropped'])
result['ranking'] = selector.ranking_



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



plt.ylim(0,1)
plt.plot(range(in_features_to_select,
               len(selector.grid_scores_) + in_features_to_select),
         selector.grid_scores_)
plt.savefig("gurafu2.png")

#allpameter

selector = RFE(rf2, n_features_to_select=6)
selector = selector.fit(X,  y)
X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])
result = pd.DataFrame(selector.get_support(), index=X.columns.values, columns=['False: dropped'])
result['ranking'] = selector.ranking_
print(X_new.columns )


Xnew_train , Xnew_test, ynew_train, ynew_test = train_test_split(X_new, y, train_size=0.6, random_state=123)


param_grid = { "n_estimators":[50,100,200,300,400],'max_depth':[10,20,30,40,50,60]}


forest_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                 param_grid = param_grid,   
                 scoring="r2",  #metrics
                 cv = 5,              #cross-validation
                 n_jobs = 1)          #number of core

forest_grid.fit(X_train,y_train) #fit
result = forest_grid.predict(X_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))


forest_grid.fit(Xnew_train, ynew_train) #fit

result = forest_grid.predict(Xnew_test)
MSE=mean_squared_error(ynew_test, result)
R2=r2_score(ynew_test,result)
print(MSE,R2)
forest_grid_best = forest_grid.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_grid.best_params_)
print('Best cross-validation: {}'.format(forest_grid.best_score_))




'''

selector = RFE(rf, n_features_to_select=6)

X_new = pd.DataFrame(selector.fit_transform(X, y), 
                     columns=X.columns.values[selector.get_support()])
result = pd.DataFrame(selector.get_support(), index=X.columns.values, columns=['False: dropped'])
result['ranking'] = selector.ranking_
print(result)



RFclf = RandomForestRegressor(n_estimators=100)

	
model = RFclf.fit(X_train, y_train)

predicted = model.predict(X_test)

print(predicted)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 
#予測値と正解値を描写する関数
def True_Pred_map(pred_df):
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    R2 = r2_score(pred_df['true'], pred_df['pred']) 
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    ax.scatter('true', 'pred', data=pred_df)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
    ax.set_ylim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
    x = np.linspace(pred_df.min().min()-0.1, pred_df.max().max()+0.1, 2)
    y = x
    ax.plot(x,y,'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)

pred_df = pd.concat([y_test.reset_index(drop=True), pd.Series(predicted)], axis=1)
pred_df.columns = ['true', 'pred']

pred_df.head()

True_Pred_map(pred_df)
plt.savefig("gurafu1.png")


RMSE_list = []
count = []
for i in range(1, 1001,100):
    RFclf = RandomForestRegressor(n_estimators=i)
    model = RFclf.fit(X_train, y_train)
    predicted = model.predict(X_test)
    pred_df = pd.concat([y_test.reset_index(drop=True), pd.Series(predicted)], axis=1)
    pred_df.columns = ['true', 'pred']
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    RMSE_list.append(RMSE)
    count.append(i)



plt.figure(figsize=(16,8))
plt.plot(count, RMSE_list, marker="o")
plt.title("RMSE Values", fontsize=30)
plt.xlabel("n_estimators", fontsize=20)
plt.ylabel("RMSE Value", fontsize=20)
plt.grid(True)
plt.savefig("gurafu2.png")

feature_importances = pd.DataFrame([X_train.columns, model.feature_importances_]).T
feature_importances.columns = ['features', 'importances']
print(feature_importances)

plt.figure(figsize=(20,10))
plt.title('Importances')
plt.rcParams['font.size']=10
sns.barplot(y=feature_importances['features'], x=feature_importances['importances'], palette='viridis')


plt.savefig("gurafu3.png")


'''

