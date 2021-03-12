import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor


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

class Maesyori:

    def __init__(self,variable1,variable2,variable3):
        self.variable1 = variable1
        self.variable2 = variable2
        self.variable3 = variable3

    def seikikavariable1(self):
        minvariable1 = min(self.variable1)
        maxvariable1 = max(self.variable1)
        self.variable1=np.array(self.variable1)
        return list(((self.variable1)-minvariable1)/(maxvariable1-minvariable1))

    def seikikavariable2(self):
        minvariable2 = min(self.variable2)
        maxvariable2 = max(self.variable2)
        self.variable2=np.array(self.variable2)
        return list(((self.variable2)-minvariable2)/(maxvariable2-minvariable2))

    def seikikavariable3(self):
        minvariable3 = min(self.variable3)
        maxvariable3 = max(self.variable3)
        self.variable3=np.array(self.variable3)
        return list(((self.variable3)-minvariable3)/(maxvariable3-minvariable3))

    def seikikavariable12(self,variable12):
        minvariable1 = min(self.variable1)
        maxvariable1 = max(self.variable1)
        variable12=np.array(variable12)
        return list((variable12-minvariable1)/(maxvariable1-minvariable1))
    
    def seikikavariable22(self,variable22):
        minvariable2 = min(self.variable2)
        maxvariable2 = max(self.variable2)
        variable22=np.array(variable22)
        return list((variable22-minvariable2)/(maxvariable2-minvariable2))
    
    def seikikavariable32(self,variable32):
        minvariable3 = min(self.variable3)
        maxvariable3 = max(self.variable3)
        variable32=np.array(variable32)
        return list((variable32-minvariable3)/(maxvariable3-minvariable3))

class gyouretu:
    def __init__(self,n):
        self.n = n
        
    def gyouretu1(self, variable1, variable2, variable3):
        d1 = {'pc[bar]':variable1,'M':variable2 ,'gamma':variable3}
        df = pd.DataFrame(d1)
        return df
    
    def gyouretu2(self, variable1, variable2, variable3,LR):
        d = {'pc[bar]':variable1 ,'M':variable2 ,'gamma':variable3,'反射点距離':LR}
        dfs = pd.DataFrame(d)
        return dfs



n=1000

df = pd.read_excel('\\CEA\\c2h2-c2h4-h2-c2h6-C2H2AR50-C2H2AR80-C2H4AR75.xlsx')



dfi=df.query('H2==1 | C2H2==1 | C2H4==1 | C2H2Ar50==1 | C2H2AR80==1 | C2H4AR75==1')


pc=dfi['pc[bar]']
M=dfi['M']
gamma=dfi['gamma']
LR=dfi['反射点距離']

dfC2H6=df.query('C2H6==1')

pcC2H6=dfC2H6['pc[bar]']
MC2H6=dfC2H6['M']
gammaC2H6=dfC2H6['gamma']
LRC2H6=dfC2H6['反射点距離']


p1=Maesyori(pc,M,gamma)
pcs=p1.seikikavariable1()
Ms=p1.seikikavariable2()
gammas=p1.seikikavariable3()

p2=gyouretu(n)
dfs=p2.gyouretu2(pcs,Ms,gammas,LR)

x = dfs.drop("反射点距離", axis=1)
y = dfs['反射点距離']
x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.8, random_state=123)
models = MLPRegressor(hidden_layer_sizes=(100,100,100,100),alpha=1e-1,solver='adam',activation='relu')
models.fit(x_trains, y_trains)
results=models.predict(x_tests)
print(mean_squared_error(y_tests, results))
print(r2_score(y_tests,results))


grafuM=MC2H6
gurafugamma=gammaC2H6
gurafupc=pcC2H6

gurafupc2=p1.seikikavariable12(gurafupc)
grafuM2=p1.seikikavariable22(grafuM)
gurafugamma2=p1.seikikavariable32(gurafugamma)
gurafux=p2.gyouretu1(gurafupc2,grafuM2,gurafugamma2)
gurafuLR1=models.predict(gurafux)
print(mean_squared_error(LRC2H6, gurafuLR1),'c2h6')
print(r2_score(LRC2H6,gurafuLR1),'c2h6')


fig = plt.figure()
plt.figure(figsize=(4,4)) #横縦
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



# グラフを描画する際の値の範囲を設定する
values =  np.concatenate([LRC2H6, gurafuLR1], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

# 出力した予測値を基にグラフを描画する
plt.scatter(y_tests, results)
#plt.scatter(LRC2H4, gurafuLR1)
#plt.scatter(LRC2H4, gurafuLR1, s=10, c="pink", alpha=0.5, linewidths="2",edgecolors="red")
plt.plot([min_value, max_value], [min_value, max_value],color='black')
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)
plt.xlabel('correct')
plt.ylabel('predict')
plt.show()

fig.savefig("img.png")







fig = plt.figure()
plt.figure(figsize=(4,4)) #横縦
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



# グラフを描画する際の値の範囲を設定する
values =  np.concatenate([LRC2H6, gurafuLR1], 0)
ptp = np.ptp(values)
min_value = np.min(values) - ptp * 0.1
max_value = np.max(values) + ptp * 0.1

# 出力した予測値を基にグラフを描画する
#plt.scatter(y_tests, results)
#plt.scatter(LRC2H4, gurafuLR1)
plt.scatter(LRC2H6, gurafuLR1, s=10, c="pink", alpha=0.5, linewidths="2",edgecolors="red")
plt.plot([min_value, max_value], [min_value, max_value],color='black')
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)
plt.xlabel('correct')
plt.ylabel('predict')
plt.show()

fig.savefig("img.png")






