import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#C2H2 O2
#a1son=[147.801 ,186.355 ,527.103 ,133.262 ,263.363 ]
#a1=[157.16193482,195.24380405,506.0792812,143.99351845,260.85222752]
a1=[147.801 ,186.355 ,527.103 ,133.262 ,263.363 ]
fai1=[1.15 ,1.0 ,0.55 ,1.3 , 0.8]
#C2H4 O2
#a2son=[784.015 ,546.57 ,542.52 ,1401.14 ,548.41 ,557.21 ]
#a2=[760.1702947566895,560.7173620342139,557.632863713739,1331.0704321449603,551.8857978467067,598.7164116111799]
a2=[784.015 ,546.57 ,542.52 ,1401.14 ,548.41 ,557.21 ]
fai2=[0.89 ,1.27 ,1.0 ,0.55 ,1.45,1.6 ]

#H2 O2
#a3son=[2253.611 ,2224.94 ,2283.07 ,2440.47 ,2285.95 ,2977.94 ,2395.81 ]
#a3=[2238.377993462328,2291.834342925457,2250.81678701269,2389.840030504452,2250.4642907522584,3048.3608072668276,2319.3691956706457,]
a3=[2253.611 ,2224.94 ,2283.07 ,2440.47 ,2285.95 ,2977.94 ,2395.81, 2642.705]
fai3=[1.0 ,0.83 ,0.95 ,1.05 ,0.63 ,1.3 ,0.55, 1.15]

#C2H6

a4=[1248.2786894978278,1882.8530868246514,1296.9292278022342,1267.6449453938626]
fai4=[1.0 ,0.7 ,1.39 ,1.2 ]

#C2H2AR50
a5=[454.7146045]
fai5=[1]

#C2H2AR80
a6=[1955.638]
fai6=[1]

#C2H4AR75
a7=[2894.671433]
fai7=[1]

def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))

df = pd.read_excel('/mnt/c/CEA/c2h2-c2h4-h2-c2h6-C2H2AR50-C2H2AR80-C2H4AR75.xlsx')

l = ['pc[bar]', 'Hc[KJ/kg]', 'Mc', 'gamma_c','p[bar]', 'T[K]', 'rho[kg/m^3]', 'H[KJ/kg]', 'G[KJ/kg]','S[KJ/kg K]',\
   'M', 'Cp[KJ/kg K]', 'gamma', 'V_CJ[m/s]']

dfs = pd.DataFrame({'pc[bar]':df[l[0]]})

for i in range(len(l)):
    params=seikika(df[l[i]])
    dfs[l[i]]=params

print(dfs)

x = dfs
y = df['反射点距離']
x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

#主成分分析の実行
pca = PCA()
pca.fit(x_trains)
# データを主成分空間に写像
feature = pca.transform(x_trains)

print(pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(x_trains.columns))]))


import matplotlib.ticker as ticker
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
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.savefig("PCA1.png")
plt.show()


plt.figure(figsize=(6,5))

plt.rcParams['font.size'] = 10
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
for x, y, name in zip(pca.components_[0], pca.components_[1], dfs.columns[0:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("PCA2.png")
plt.show()




plt.figure(figsize=(6,5))

plt.rcParams['font.size'] = 10
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
for x, y, name in zip(pca.components_[2], pca.components_[3], dfs.columns[0:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[2], pca.components_[3], alpha=0.8)
plt.xlabel("PC3")
plt.ylabel("PC4")
plt.savefig("PCA3.png")
plt.show()



x = dfs
y = df['反射点距離']

pca = PCA(n_components=2)

x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = pca.fit_transform(x_trains)

pca_x_tests = pca.transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),2)
print(r2_score(y_tests,results),2)


pca = PCA(n_components=3)


x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = pca.fit_transform(x_trains)

pca_x_tests = pca.transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),3)
print(r2_score(y_tests,results),3)



pca = PCA(n_components=4)

x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = pca.fit_transform(x_trains)

pca_x_tests = pca.transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),4)
print(r2_score(y_tests,results),4)


'''
x = dfs
y = df['反射点距離']

kpca = KernelPCA(n_components=2,  kernel='rbf')

x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = kpca.fit_transform(x_trains)

pca_x_tests = kpca.fit_transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),2)
print(r2_score(y_tests,results),2)


x = dfs
y = df['反射点距離']

kpca = KernelPCA(n_components=2,  kernel='rbf')

x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = kpca.fit_transform(x_trains)

pca_x_tests = kpca.fit_transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),2)
print(r2_score(y_tests,results),2)


x = dfs
y = df['反射点距離']

kpca = KernelPCA(n_components=3,  kernel='rbf')

x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = kpca.fit_transform(x_trains)

pca_x_tests = kpca.fit_transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),3)
print(r2_score(y_tests,results),3)


x = dfs
y = df['反射点距離']

kpca = KernelPCA(n_components=4,  kernel='rbf')

x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)

pca_x_trains = kpca.fit_transform(x_trains)

pca_x_tests = kpca.fit_transform(x_tests)

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(pca_x_trains, y_trains)
results=models.predict(pca_x_tests)
print(mean_squared_error(y_tests, results),4)
print(r2_score(y_tests,results),4)
'''

'''
# モデル(パイプライン)の作成
clf = Pipeline(
    [
        ("pca", PCA()),
        ("lr", MLPRegressor())
    ]
)

# 探索するパラメータの設定
# ここのkeyの指定方法が重要
params = {
    "pca__n_components": [2, 3, 4],
    "lr__penalty": ["l1", "l2"],
    "lr__C": [0.01, 0.1, 1, 10, 100]
}

# グリッドサーチ
gs_clf = GridSearchCV(
    clf,
    params,
    cv=5
)
gs_clf.fit(X_train, y_train)

# 最適なパラメーター
print(gs_clf.best_params_)

# 最適なモデルの評価
best_clf = gs_clf.best_estimator_
print(classification_report(y_test, best_clf.predict(X_test)))

models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(x_trains, y_trains)
results=models.predict(x_tests)
print(mean_squared_error(y_tests, results))
print(r2_score(y_tests,results))




pca = PCA(n_components=0.95)
feature = pca.fit_transform(x_trains)

pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(x_trains.columns))])

import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.savefig("PCA.png")
plt.show()

'''