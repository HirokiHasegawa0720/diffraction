import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
import itertools
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))

df = pd.read_excel('all_mixture.xlsx')

output=df['LR'] #output= Coefficienta or LR

l=['p0[bar]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0', 'Pvn[bar]', 'Tvn[K]','Hvn[KJ/kg]', 'T_CJ[K]', 'M_CJkg/kmol', 'gamma_CJ']

for i in range(len(l)):   #seikika
    params=seikika(df[l[i]])
    df[l[i]]=params


n=100
nn=40

class neural:

    def __init__(self,n):
        self.n =n
        
    def traindataxy(self,dfs,a):
        dfs = dfs.drop(range(a*self.n,(a+1)*self.n))
        x_train = dfs.drop("反射点距離ora", axis=1)
        y_train = dfs['反射点距離ora']
        return x_train , y_train

    def testdataxy(self,dfs,a):
        dfs = dfs[a*self.n:(a+1)*self.n]
        x_test = dfs.drop("反射点距離ora", axis=1)
        y_test = dfs['反射点距離ora']
        return x_test , y_test 

    def tranning(self,l,nn,df,output):
        l1=list(itertools.combinations(l,nn))
        print(len(l1))
        if nn == 2:
            d = {'物理量1':['1'] ,'物理量2':['i'] , 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
        if nn == 3:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'], 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
        if nn == 4:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'], 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
        if nn == 5:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'], 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
        if nn == 6:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'], '物理量6':['1'],'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
        dfss = pd.DataFrame(d)

        for i in range(len(l1)):
            if nn == 2:
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                d = {ll[0]:aa ,ll[1]:bb ,'反射点距離ora':output}
            if nn == 3:
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,'反射点距離ora':output}
            if nn == 4:
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                ee=df[ll[4]]
                ff=df[ll[5]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,'反射点距離ora':output}
            if nn == 5:
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                ee=df[ll[4]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,'反射点距離ora':output}
            if nn == 6:
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                ee=df[ll[4]]
                ff=df[ll[5]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,ll[5]:ff,'反射点距離ora':output}
            dfs = pd.DataFrame(d)
            AA=[]
            BB=[]
            for j in [4,35,39]:
                x_trains = self.traindataxy(dfs,j)[0]
                y_trains = self.traindataxy(dfs,j)[1]
                x_tests = self.testdataxy(dfs,j)[0]
                y_tests = self.testdataxy(dfs,j)[1]
                models = MLPRegressor(hidden_layer_sizes=(300, 300, 300, 300, 300),alpha=1e-4,solver='adam',activation='relu')
                models.fit(x_trains, y_trains)
                results=models.predict(x_tests)
                AA.append(mean_squared_error(y_tests, results))
                BB.append(r2_score(y_tests,results))
            x = dfs.drop("反射点距離ora", axis=1)
            y = dfs['反射点距離ora']
            x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
            models = MLPRegressor(hidden_layer_sizes=(300, 300, 300, 300, 300),alpha=1e-4,solver='adam',activation='relu')
            models.fit(x_trains, y_trains)
            results=models.predict(x_tests)
            A=mean_squared_error(y_tests, results)
            B=r2_score(y_tests,results)
            if nn == 2:
                dfss.loc[i]=[ll[0],ll[1],A,B,AA[0],BB[0],AA[1],BB[1],AA[2],BB[2]]
            if nn == 3:
                dfss.loc[i]=[ll[0],ll[1],ll[2],A,B,AA[0],BB[0],AA[1],BB[1],AA[2],BB[2]]
            if nn == 4:
                dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],A,B,AA[0],BB[0],AA[1],BB[1],AA[2],BB[2]]
            if nn == 5:
                dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],A,B,AA[0],BB[0],AA[1],BB[1],AA[2],BB[2]]
            if nn == 6:
                dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],ll[5],A,B,AA[0],BB[0],AA[1],BB[1],AA[2],BB[2]]


        dfss=dfss.sort_values('R2', ascending=False)
        return dfss



p1 = neural(n)
df2 = p1.tranning(l,2,df,output)
df2.to_excel('/mnt/c/CEA/params2.xlsx')

df3 = p1.tranning(l,3,df,output)
df3.to_excel('/mnt/c/CEA/params3.xlsx')

df4 = p1.tranning(l,4,df,output)
df4.to_excel('/mnt/c/CEA/params4.xlsx')

df5 = p1.tranning(l,5,df,output)
df5.to_excel('/mnt/c/CEA/params5.xlsx')

df6 = p1.tranning(l,6,df,output)
df2.to_excel('/mnt/c/CEA/params6.xlsx')

