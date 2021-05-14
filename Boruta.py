import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns

def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))

df = pd.read_excel('/mnt/c/CEA/all_mixture.xlsx')

output=df['LR'] #output= Coefficienta or LR

l=['p0[bar]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0','p_CJ[bar]', 'T_CJ[K]', 'H_CJ[KJ/kg]', 'gamma_CJ','M_CJ', 'Tvn[K]']

for i in range(len(l)):   #seikika
    params=seikika(df[l[i]])
    df[l[i]]=params


dfs = pd.DataFrame({'p0[bar]':df['p0[bar]']})

for i in range(len(l)):
    params=df[l[i]]
    dfs[l[i]]=params


X = dfs
y = output
X, X_test, y, y_test = train_test_split(X, y, train_size=0.4, random_state=123)

'''
# 全部の特徴量で学習
rf1 = RandomForestRegressor(n_jobs=-1, max_depth=5)
rf1.fit(X, y)
print('SCORE with ALL Features: %1.2f\n' % rf1.score(X, y))

# RandomForestRegressorでBorutaを実行
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X.values, y.values)

# 選択された特徴量を確認
selected = feat_selector.support_
print('選択された特徴量の数: %d' % np.sum(selected))
print(selected)
print(X.columns[selected])

# 選択した特徴量で学習
X_selected = X[X.columns[selected]]
rf2 = RandomForestRegressor(n_jobs=-1, max_depth=5)
rf2.fit(X_selected, y)
print('SCORE with selected Features: %1.2f' % rf2.score(X_selected, y))
'''
X_train , X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, random_state=123)

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
