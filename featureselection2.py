import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
import itertools
from sklearn.model_selection import GridSearchCV

def seikika(h):
    minh = min(h)
    maxh = max(h)
    h1=np.array(h)
    return list(((h1)-minh)/(maxh-minh))

df = pd.read_excel('all_mixture1.xlsx')

output=df['LR'] #output= Coefficienta or LR

l=['p0[bar]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0','p_CJ[bar]', 'T_CJ[K]', 'H_CJ[KJ/kg]', 'M_CJkg/kmol', 'gamma_CJ','M_CJ', 'Tvn[K]']


for i in range(len(l)):   #seikika
    params=seikika(df[l[i]])
    df[l[i]]=params

n=100

class neural:

    def __init__(self,n):
        self.n =n

    def testdataxy(self,test,a):
        test = test[a*self.n:(a+1)*self.n]
        return test

    def tranning(self,l,nn,df,output):
        l1=list(itertools.combinations(l,nn))
        print(len(l1))
        if nn == 2:
            d = {'物理量1':['1'] ,'物理量2':['i'] , 'MSE':[10000],'R2':[0],'Bestparams':[0],'indexMSE':[0],'indexR2':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' , 'MSE','R2','Bestparams','MSEindex','R2index','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O'])
        if nn == 3:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'], 'MSE':[10000],'R2':[0],'Bestparams':[0],'indexMSE':[0],'indexR2':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3', 'MSE','R2','Bestparams','MSEindex','R2index','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O'])
        if nn == 4:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'], 'MSE':[10000],'R2':[0],'Bestparams':[0],'indexMSE':[0],'indexR2':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3','物理量4', 'MSE','R2','Bestparams','MSEindex','R2index','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O'])
        if nn == 5:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'], 'MSE':[10000],'R2':[0],'Bestparams':[0],'indexMSE':[0],'indexR2':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3','物理量4','物理量5', 'MSE','R2','Bestparams','MSEindex','R2index','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O'])
        if nn == 6:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'], '物理量6':['1'],'MSE':[10000],'R2':[0],'Bestparams':[0],'indexMSE':[0],'indexR2':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0],'H2':[0],'C2H2':[0],'C2H4':[0],'C2H6':[0],'C3H6':[0],'C2H2AR':[0],'C2H4AR':[0],'C2H2N2':[0],'C2H2He':[0],'C2H2Kr':[0],'C2H2N2O':[0],'C3H6N2O':[0]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3','物理量4','物理量5','物理量6', 'MSE','R2','Bestparams','MSEindex','R2index','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O','H2','C2H2','C2H4','C2H6','C3H6','C2H2AR','C2H4AR','C2H2N2','C2H2He','C2H2Kr','C2H2N2O','C3H6N2O'])

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
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,'反射点距離ora':output}
            if nn == 5:
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                ee=df[ll[4]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,'反射点距離ora':output}
            if nn == 6:
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                ee=df[ll[4]]
                ff=df[ll[5]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,ll[5]:ff,'反射点距離ora':output}
            dfs = pd.DataFrame(d)
            x = dfs.drop("反射点距離ora", axis=1)
            y = dfs['反射点距離ora']
            AA=[]
            BB=[]
            bestparms=[]
            x_trains, x_tests, y_trains, y_tests = train_test_split(x, y, train_size=0.6, random_state=123)
            sol=['adam']
            act=['relu']
            hidd=[]
            for i in [2,4,6]:
                for j in [100,200,300]:
                    b=[j]*i
                    b=tuple(b)
                    hidd.append(b)
            alp=[ 1e-7,1e-4,1e-1]
            param_grid = {'solver':sol,'activation':act,'hidden_layer_sizes':hidd,'alpha':alp}
            grid = GridSearchCV(MLPRegressor(),param_grid,cv=3)
            grid.fit(x_trains,y_trains)
            for j in [4,10,15,22,27,30,32,35,36,37,38,39]:
                x_tests = self.testdataxy(x_tests,j)
                y_tests = self.testdataxy(y_tests,j)
                results=grid.predict(x_tests)
                MSE=mean_squared_error(y_tests, results)
                R2=r2_score(y_tests,results)
                AA.append(MSE)
                BB.append(R2)
                bestparms.append(grid.best_params_)
            A=max(AA)
            B=min(BB)
            AAA=AA.index(max(AA))
            BBB=BB.index(min(BB))
            bastparam=bestparms[BB.index(min(BB))]
            print(bastparam)
            if nn == 2:
                dfss.loc[i]=[ll[0],ll[1],A,B,bastparam,AAA,BBB,AA[0],AA[1],AA[2],AA[3],AA[4],AA[5],AA[6],AA[7],AA[8],AA[9],AA[10],AA[11],BB[0],BB[1],BB[2],BB[3],BB[4],BB[5],BB[6],BB[7],BB[8],BB[9],BB[10],BB[11]]
            if nn == 3:
                dfss.loc[i]=[ll[0],ll[1],ll[2],A,B,bastparam,AAA,BBB,AA[0],AA[1],AA[2],AA[3],AA[4],AA[5],AA[6],AA[7],AA[8],AA[9],AA[10],AA[11],BB[0],BB[1],BB[2],BB[3],BB[4],BB[5],BB[6],BB[7],BB[8],BB[9],BB[10],BB[11]]
            if nn == 4:
                dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],A,B,bastparam,AAA,BBB,AA[0],AA[1],AA[2],AA[3],AA[4],AA[5],AA[6],AA[7],AA[8],AA[9],AA[10],AA[11],BB[0],BB[1],BB[2],BB[3],BB[4],BB[5],BB[6],BB[7],BB[8],BB[9],BB[10],BB[11]]
            if nn == 5:
                dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],A,B,bastparam,AAA,BBB,AA[0],AA[1],AA[2],AA[3],AA[4],AA[5],AA[6],AA[7],AA[8],AA[9],AA[10],AA[11],BB[0],BB[1],BB[2],BB[3],BB[4],BB[5],BB[6],BB[7],BB[8],BB[9],BB[10],BB[11]]
            if nn == 6:
                dfss.loc[i]=[ll[0],ll[1],ll[2],ll[3],ll[4],ll[5],A,B,bastparam,AAA,BBB,AA[0],AA[1],AA[2],AA[3],AA[4],AA[5],AA[6],AA[7],AA[8],AA[9],AA[10],AA[11],BB[0],BB[1],BB[2],BB[3],BB[4],BB[5],BB[6],BB[7],BB[8],BB[9],BB[10],BB[11]]


        dfss=dfss.sort_values('R2', ascending=False)
        return dfss



p1 = neural(n)


df2 = p1.tranning(l,2,df,output)
df2.to_excel('params2.xlsx')

df3 = p1.tranning(l,3,df,output)
df3.to_excel('params3.xlsx')

df4 = p1.tranning(l,4,df,output)
df4.to_excel('params4.xlsx')

df5 = p1.tranning(l,5,df,output)
df5.to_excel('params5.xlsx')

df6 = p1.tranning(l,6,df,output)
df6.to_excel('params6.xlsx')


'''

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
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' , 'MSE','R2','H2/O2/MSE', 'H2/O2/R2','C2H4/N2MSE','C2H4/N2R2','C3H6/N2OMSE','C3H6/N2OR2'])
        if nn == 3:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'], 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3', 'MSE','R2','H2/O2/MSE', 'H2/O2/R2','C2H4/N2MSE','C2H4/N2R2','C3H6/N2OMSE','C3H6/N2OR2'])
        if nn == 4:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'], 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3','物理量4', 'MSE','R2','H2/O2/MSE', 'H2/O2/R2','C2H4/N2MSE','C2H4/N2R2','C3H6/N2OMSE','C3H6/N2OR2'])
        if nn == 5:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'], 'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3','物理量4','物理量5', 'MSE','R2','H2/O2/MSE', 'H2/O2/R2','C2H4/N2MSE','C2H4/N2R2','C3H6/N2OMSE','C3H6/N2OR2'])
        if nn == 6:
            d = {'物理量1':['1'] ,'物理量2':['i'] ,'物理量3':['1'],'物理量4':['1'],'物理量5':['1'], '物理量6':['1'],'MSE':[10000],'R2':[0],'H2/O2/MSE':[6], 'H2/O2/R2':[6],'C2H4/N2MSE':[37],'C2H4/N2R2':[37],'C3H6/N2OMSE':[41],'C3H6/N2OR2':[41]}
            dfss = pd.DataFrame(d,columns=['物理量1' ,'物理量2' ,'物理量3','物理量4','物理量5','物理量6', 'MSE','R2','H2/O2/MSE', 'H2/O2/R2','C2H4/N2MSE','C2H4/N2R2','C3H6/N2OMSE','C3H6/N2OR2'])

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
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,'反射点距離ora':output}
            if nn == 5:
                ll=l1[i]
                aa=df[ll[0]]
                bb=df[ll[1]]
                cc=df[ll[2]]
                dd=df[ll[3]]
                ee=df[ll[4]]
                d = {ll[0]:aa ,ll[1]:bb ,ll[2]:cc,ll[3]:dd,ll[4]:ee,'反射点距離ora':output}
            if nn == 6:
                ll=l1[i]
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

'''

