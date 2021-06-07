import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

# California Housingデータセット
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['Price'] = housing.target
x = df_housing.drop(['Price'], axis=1)
y = df_housing['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
df_housing.head()

ss = StandardScaler()
 
sx_train = ss.fit_transform(x_train)#正規化

sx_test = ss.transform(x_test) # fit_transformじゃなくtransformであることに注意（学習のときの最大値，最小値を使って正規化）


sx_train = pd.DataFrame(sx_train, columns=housing.feature_names) #patial dependence plot のためdetaframeに戻す
sx_train


model=MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-4,solver='adam',activation='relu') #計算コストの都合上gridsearchしていません
model.fit(sx_train,y_train)

result=model.predict(sx_test)
MSE=mean_squared_error(y_test, result)
R2=r2_score(y_test,result)
print(MSE,R2)


def plot_feature_importance(df):
    n_features = len(df)
    df_plot = df.sort_values('importance')
    f_importance_plot = df_plot['importance'].values
    plt.barh(range(n_features), f_importance_plot, align='center')
    cols_plot = df_plot['feature'].values
    plt.yticks(np.arange(n_features), cols_plot)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')


result = permutation_importance(model,sx_train,y_train, n_repeats=5, random_state=42)

cols = list(sx_train.columns)         # 特徴量名のリスト(目的変数以外)
f_importance = np.array(result["importances"].mean(axis=1)) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values("importance",ascending=False)

plt.figure()
plot_feature_importance(df_importance)
plt.title("Permutation Importance(mlp)")
plt.tight_layout()
plt.show()

features = ["Longitude", "Latitude", "MedInc"]
plt.figure()
plot_partial_dependence(estimator=model, X=sx_train, features=features,
                        n_jobs=-4, grid_resolution=20, n_cols=2)
plt.show()