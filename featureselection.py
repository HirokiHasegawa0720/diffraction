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

df = pd.read_excel('/mnt/c/CEA/all_mixture.xlsx')

'''
#相関係数0.9以上の特徴量の削除

l=['p0[bar]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0', 'SonicVelocity_0[m/s]', \
    'U0[kg/kj]','SonicVelocity0[m/s]',\
        'p_CJ[bar]', 'T_CJ[K]', 'H_CJ[KJ/kg]', 'U_CJ[KJ/kg]',\
                    'M_CJkg/kmol', 'gamma_CJ', 'SonicVelocity_CJ[m/s]',\
                         'M_CJ', 'V_CJ[m/s]',\
        'Uvn[m/s]','Pvn[bar]', 'Tvn[K]', 'Hvn[KJ/kg]', 'Uvn[KJ/kg]',\
                      'gammavn','SonicVelocityvn[m/s]', \
                        ]


dfs = pd.DataFrame({'p0[bar]':df[l[0]]})

for i in range(len(l)):
    params=df[l[i]]
    dfs[l[i]]=params


x = dfs
y = df['LR']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=123)

dfss=dfs
dfss['reflection']=df['LR']

g = sns.pairplot(dfss)
plt.savefig("sns.png")

threshold = 0.9

feat_corr = set()
corr_matrix = X_train.corr()

corr_matrix.to_excel('/mnt/c/CEA/matrix4.xlsx')

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
        

print(corr_matrix)
X_train.drop(labels=feat_corr, axis='columns', inplace=True)
X_test.drop(labels=feat_corr, axis='columns', inplace=True)
print(X_train.columns)
print(len(X_train.columns))


corr_matrix.to_excel('/mnt/c/CEA/matrix3.xlsx')
'''

'''
MI = mutual_info_regression(X_train, y_train)
MI = pd.Series(MI)
MI.index = X_train.columns
print(MI)

mi_reg = pd.Series(mutual_info_regression(X_train, y_train),index=X_train.columns).sort_values(ascending=False)
print(mi_reg)
X = X_train.loc[:, mi_reg.iloc[:8].index]
print(X)
g = sns.pairplot(X.assign(reflection = y_train), y_vars=['reflection'], x_vars=X.columns)
sns.despine()
plt.savefig("mutual.png")

#MI.sort_values(ascending=False).plot(kind='bar', figsize=(20,10))

# KBest : 抽出する特徴量の"数"を指定
kbest_sel_ = SelectKBest(mutual_info_regression, k=5)
print(len(kbest_sel_.get_support()))
print(kbest_sel_)
# Percentile : 抽出する特徴量の割合を指定
percentile_sel_ = SelectPercentile(mutual_info_regression, percentile=10)
print(len(percentile_sel_.get_support()))
print(percentile_sel_)

'''


#機械学習の結果の解析

df2 = pd.read_excel('/mnt/c/CEA/params2.xlsx')
df3 = pd.read_excel('/mnt/c/CEA/params3.xlsx')
df4 = pd.read_excel('/mnt/c/CEA/params4.xlsx')
df5 = pd.read_excel('/mnt/c/CEA/params5.xlsx')
df6 = pd.read_excel('/mnt/c/CEA/params6.xlsx')

l=['p0[bar]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0','p_CJ[bar]', 'T_CJ[K]', 'H_CJ[KJ/kg]', 'gamma_CJ','M_CJ', 'Tvn[K]']

probability=[]


dfs = pd.DataFrame({'pc[bar]':[1,1,1,1]})

n=0.8

for i in range(len(l)):
    df21=df2[df2['R2'] > n]
    df22=df2[df2['R2'] < n]
    df23=df21[(df21['物理量1'] == l[i]) | (df21['物理量2'] == l[i])]
    df24=df22[(df22['物理量1'] == l[i]) | (df22['物理量2'] == l[i])]
    params2=(len(df23)/(len(df23)+len(df24)))
    df31=df3[df3['R2'] > n]
    df32=df3[df3['R2'] < n]
    df33=df31[(df31['物理量1'] == l[i]) | (df31['物理量2'] == l[i]) | (df31['物理量3'] == l[i])]
    df34=df32[(df32['物理量1'] == l[i]) | (df32['物理量2'] == l[i]) | (df32['物理量3'] == l[i])]
    params3=(len(df33)/(len(df33)+len(df34)))
    df41=df4[df4['R2'] > n]
    df42=df4[df4['R2'] < n]
    df43=df41[(df41['物理量1'] == l[i]) | (df41['物理量2'] == l[i]) | (df41['物理量3'] == l[i]) | (df41['物理量4'] == l[i])]
    df44=df42[(df42['物理量1'] == l[i]) | (df42['物理量2'] == l[i]) | (df42['物理量3'] == l[i]) | (df42['物理量4'] == l[i])]
    params4=(len(df43)/(len(df43)+len(df44)))
    df51=df5[df5['R2'] > n]
    df52=df5[df5['R2'] < n]
    df53=df51[(df51['物理量1'] == l[i]) | (df51['物理量2'] == l[i]) | (df51['物理量3'] == l[i]) | (df51['物理量4'] == l[i]) | (df51['物理量5'] == l[i])]
    df54=df52[(df52['物理量1'] == l[i]) | (df52['物理量2'] == l[i]) | (df52['物理量3'] == l[i]) | (df52['物理量4'] == l[i]) | (df52['物理量5'] == l[i])]
    params5=(len(df53)/(len(df53)+len(df54)))
    '''
    df61=df6[df6['R2'] > 0.95]
    df62=df6[df6['R2'] < 0.95]
    df63=df61[(df61['物理量1'] == l[i]) | (df61['物理量2'] == l[i]) | (df61['物理量3'] == l[i]) | (df61['物理量4'] == l[i]) | (df61['物理量5'] == l[i]) | (df61['物理量6'] == l[i])]
    df64=df62[(df62['物理量1'] == l[i]) | (df62['物理量2'] == l[i]) | (df62['物理量3'] == l[i]) | (df62['物理量4'] == l[i]) | (df62['物理量5'] == l[i]) | (df62['物理量6'] == l[i])]
    params6=(len(df63)/(len(df63)+len(df64)))
    '''
    
    probability=[params2,params3,params4,params5]
    dfs[l[i]]=probability
    
dfs.to_excel('/mnt/c/CEA/probability.xlsx')
x2=[]
x3=[]
x4=[]
x5=[]
x6=[]

for i in range(len(df2['R2'])):
    x2.append(i)

for i in range(len(df2['R2']),len(df2['R2'])+len(df3['R2']),1):
    x3.append(i)

for i in range(len(df2['R2'])+len(df3['R2']),len(df2['R2'])+len(df3['R2'])+len(df4['R2']),1):
    x4.append(i)

for i in range(len(df2['R2'])+len(df3['R2'])+len(df4['R2']),len(df2['R2'])+len(df3['R2'])+len(df4['R2'])+len(df5['R2']),1):
    x5.append(i)

#for i in range(len(df2['R2'])+len(df3['R2'])+len(df4['R2'])+len(df5['R2']),len(df2['R2'])+len(df3['R2'])+len(df4['R2'])+len(df5['R2'])+len(df6['R2']),1):
#    x6.append(i)

y2=list(df2['R2'])
for i in range(len(y2)):
    y2[i]=math.log(1/(1-y2[i]))
y3=list(df3['R2'])
for i in range(len(y3)):
    y3[i]=math.log(1/(1-y3[i]))
y4=list(df4['R2'])
for i in range(len(y4)):
    y4[i]=math.log(1/(1-y4[i]))
y5=list(df5['R2'])
for i in range(len(y5)):
    y5[i]=math.log(1/(1-y5[i]))
#y6=list(df6['R2'])


fig = plt.figure()
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
plt.plot(x2, y2,color='blue')
plt.plot(x3, y3,color='red')
plt.plot(x4, y4,color='black')
plt.plot(x5, y5,color='green')
#plt.plot(x6, y6,color='cyan')
plt.xlabel("parameter number[-]", fontsize=20)
plt.ylabel("log(1/(1-R2))[-]", fontsize=20)
plt.savefig("R2.png")
plt.show()
y2=[]
y3=[]
y4=[]
y5=[]
y2=list(df2['R2'])
y3=list(df3['R2'])
y4=list(df4['R2'])
y5=list(df5['R2'])
plt.ylim(0.9, 1.00)
plt.plot(x2, y2,color='blue')
plt.plot(x3, y3,color='red')
plt.plot(x4, y4,color='black')
plt.plot(x5, y5,color='green')
#plt.plot(x6, y6,color='cyan')
plt.xlabel("parameter number[-]", fontsize=20)
plt.ylabel("log(1/(1-R2))[-]", fontsize=20)
plt.savefig("R21.png")
plt.show()



#RMSE


df2=df2.sort_values('MSE')
df3=df3.sort_values('MSE')
df4=df4.sort_values('MSE')
df5=df5.sort_values('MSE')

x2=[]
x3=[]
x4=[]
x5=[]


for i in range(len(df2['MSE'])):
    x2.append(i)

for i in range(len(df2['MSE']),len(df2['MSE'])+len(df3['MSE']),1):
    x3.append(i)

for i in range(len(df2['MSE'])+len(df3['MSE']),len(df2['MSE'])+len(df3['MSE'])+len(df4['MSE']),1):
    x4.append(i) 

for i in range(len(df2['MSE'])+len(df3['MSE'])+len(df4['MSE']),len(df2['MSE'])+len(df3['MSE'])+len(df4['MSE'])+len(df5['MSE']),1):
    x5.append(i)

#for i in range(len(df2['MSE'])+len(df3['MSE'])+len(df4['MSE'])+len(df5['MSE']),len(df2['MSE'])+len(df3['MSE'])+len(df4['MSE'])+len(df5['MSE'])+len(df6['MSE']),1):
#    x6.append(i)

y2=list(df2['MSE'])
y3=list(df3['MSE'])
y4=list(df4['MSE'])
y5=list(df5['MSE'])
#y6=list(df6['MSE'])

for i in range(len(y2)):
    y2[i] = math.log(y2[i])

for i in range(len(y3)):
    y3[i] = math.log(y3[i])
    
for i in range(len(y4)):
    y4[i] = math.log(y4[i])

for i in range(len(y5)):
    y5[i] = math.log(y5[i])

#for i in range(len(y6)):
#    y6[i] = math.log(y6[i])

fig = plt.figure()
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

plt.plot(x2, y2,color='blue')
plt.plot(x3, y3,color='red')
plt.plot(x4, y4,color='black')
plt.plot(x5, y5,color='green')
#plt.plot(x6, y6,color='cyan')
plt.xlabel("parameter number[-]", fontsize=20)
plt.ylabel("log(MSE)[$mm^{2}$]", fontsize=20)
plt.savefig("MSE.png")
plt.show()

