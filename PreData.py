import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.optimize import curve_fit
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

#p1
class Predata:

    def __init__(self, a, fai,n):
        self.a = a
        self.fai = fai
        self.n = n
    
    def lr(self,x,a):
        return (a)*x**(-1)

    def manyFAI(self):
        FAI=[]
        for i in range(len(self.a)):
            for j in range(self.n):
                FAI.append(self.fai[i])
        return FAI
    
    def manyPLR(self):
        P=[]
        LR=[]
        for i in range(len(self.a)):
            for j in range(self.n):
                randomp = random.uniform(7,100)
                P.append(randomp)
        for i in range(len(self.a)):
            for j in range(self.n):
                LR.append(self.lr(P[j+(self.n)*i],self.a[i]))
        return P,LR

class Splitjoin:

    def __init__(self,n):
        self.n = n

    def ketugou3(self,P,FAI,C2H2,C2H4,H2,C2H6,C2H2Ar50,C2H4AR75,C2H2AR80,LR):
        d = {'初期圧力':P ,'等量比':FAI ,'C2H2':C2H2,'C2H4':C2H4,'H2':H2,'C2H6':C2H6,'C2H2Ar50':C2H2Ar50,'C2H2AR80':C2H2AR80,'C2H4AR75':C2H4AR75,'反射点距離':LR}
        df1 = pd.DataFrame(d)
        return df1



n=1000

p11 = Predata(a1,fai1,n)
FAI11 = p11.manyFAI()
PLR1 = p11.manyPLR()
P11 = PLR1[0]
LR1 = PLR1[1]

p12 = Predata(a2,fai2,n)
FAI12 = p12.manyFAI()
PLR2 = p12.manyPLR()
P12 = PLR2[0]
LR2 = PLR2[1]

p13 = Predata(a3,fai3,n)
FAI13 = p13.manyFAI()
PLR3 = p13.manyPLR()
P13 = PLR3[0]
LR3 = PLR3[1]

p14 = Predata(a4,fai4,n)
FAI14 = p14.manyFAI()
PLR4 = p14.manyPLR()
P14 = PLR4[0]
LR4 = PLR4[1]

p15 = Predata(a5,fai5,n)
FAI15 = p15.manyFAI()
PLR5 = p15.manyPLR()
P15 = PLR5[0]
LR5 = PLR5[1]

p16 = Predata(a6,fai6,n)
FAI16 = p16.manyFAI()
PLR6 = p16.manyPLR()
P16 = PLR6[0]
LR6 = PLR6[1]

p17 = Predata(a7,fai7,n)
FAI17 = p17.manyFAI()
PLR7 = p17.manyPLR()
P17 = PLR7[0]
LR7 = PLR7[1]

a10=[0]*n*len(a1)
a11=[1]*n*len(a1)
a20=[0]*n*len(a2)
a21=[1]*n*len(a2)
a30=[0]*n*len(a3)
a31=[1]*n*len(a3)
a40=[0]*n*len(a4)
a41=[1]*n*len(a4)
a50=[0]*n*len(a5)
a51=[1]*n*len(a5)
a60=[0]*n*len(a6)
a61=[1]*n*len(a6)
a70=[0]*n*len(a7)
a71=[1]*n*len(a7)

C2H2=a11+a20+a30+a40+a50+a60+a70
C2H4=a10+a21+a30+a40+a50+a60+a70
H2=a10+a20+a31+a40+a50+a60+a70
C2H6=a10+a20+a30+a41+a50+a60+a70
C2H2AR50=a10+a20+a30+a40+a51+a60+a70
C2H2AR80=a10+a20+a30+a40+a50+a61+a70
C2H4AR75=a10+a20+a30+a40+a50+a60+a71
P1=P11+P12+P13+P14+P15+P16+P17
FAI1=FAI11+FAI12+FAI13+FAI14+FAI15+FAI16+FAI17
LR=LR1+LR2+LR3+LR4+LR5+LR6+LR7


p3 = Splitjoin(n)

#kPa to bar
def calc_double(n):
    return n/100
P1=list(map(calc_double,P1))

df = p3.ketugou3(P1,FAI1,C2H2,C2H4,H2,C2H6,C2H2AR50,C2H4AR75,C2H2AR80,LR)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import subprocess
import re


#cea_path = "C:\CEA\cea-exec"    # PATH to NASA-CEA
#fcea2m = "C:\CEA\cea-exec\FCEA2m.exe" # fcea2m.exe

cea_path = "/mnt/c/CEA/cea-exec"    # PATH to NASA-CEA
fcea2m = "/mnt/c/CEA/cea-exec/FCEA2m.exe" # fcea2m.exe


class NASAcea():
    """
    This class runs NASA-CEA in python. Calculation results return as pandas series.
    I made this program to calculate C-J detonation parameter. So please change program accordingly if you want to calculate other parameters.
    If you want to use this program as module, please inform me(Tomoyuki Noda).
    """

    def det_inp(self):
        """Create .inp file (detonation)"""

        fuel = self.mixtures
        oxidizer = 'O2'
        if self.AR == 0:
            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n\n"
                        "reactant\n"
                        "    fuel = %s  t(k) = %.2f\n"
                        "    oxid = %s   t(k) = %.2f\n"
                        "outp\n"
                        "    short\n"
                        "end\n" )  % (self.Tc, self.ER, self.pc, fuel, self.Tc, oxidizer, self.Tc)
        if self.AR == 40:
            diluent='Ar'
            print(40)

            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n\n\n\n\n\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n"
                        "reactant\n"
                        "    fuel = %s  moles = %.3f\n"
                        "    fuel = %s  moles = %.3f\n"
                        "    oxid = %s  moles = %.3f\n"
                        "outp\n"
                        "    short\n"
                        "end\n" )  % (self.Tc,self.ER, self.pc, fuel, 7.6923, diluent ,92.307, oxidizer, 100 )

        if self.AR == 30:
            diluent='Ar'
            print(30)

            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n\n\n\n\n\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n"
                        "reactant\n"
                        "    fuel = %s  moles = %.3f\n"
                        "    fuel = %s  moles = %.3f\n"
                        "    oxid = %s  moles = %.3f\n"
                        "outp\n"
                        "    short\n"
                        "end\n" )  % (self.Tc,self.ER, self.pc, fuel, 6.6666, diluent ,93.333, oxidizer, 100 )
        
        if self.AR == 20:
            diluent='Ar'
            print(20)

            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n\n\n\n\n\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n"
                        "reactant\n"
                        "    fuel = %s  moles = %.3f\n"
                        "    fuel = %s  moles = %.3f\n"
                        "    oxid = %s  moles = %.3f\n"
                        "outp\n"
                        "    short\n"
                        "end\n" )  % (self.Tc,self.ER, self.pc, fuel, 22.222, diluent ,77.777, oxidizer, 100 )



        input_file = self.file_name + '.inp'
        f_in = open(input_file, 'w')
        f_in.write(setting)
        f_in.close()


    

    def det_out(self):
        """Read .out file and convert pandas data (detonation)"""
        output_file = self.file_name + '.out'
        f_out = open(output_file, 'r')
        lines = f_out.readlines()
        f_out.close()

        unb = np.zeros(6)
        bnd = np.zeros(13)
        dpr = np.zeros(6)

        for i in range(6):
            word = re.split(" +", lines[i+43])
            unb[i] = float(word[-1].strip())

        for i in range(2):
            word = re.split(" +", lines[i+52])
            bnd[i] = float(word[-1].strip())
        word = re.split(" +", lines[54])
        if len(word) == 6:
            bnd[2] = float(word[-2]) * 10**float(word[-1].strip())
        else:
            num = word[-1].split('-')
            bnd[2] = float(num[0]) * 10**float(num[1].strip())
        for i in range(4):
            word = re.split(" +", lines[i+55])
            bnd[i+3] = float(word[-1].strip())
        for i in range(6):
            word = re.split(" +", lines[i+60])
            bnd[i+7] = float(word[-1].strip())

        for i in range(6):
            word = re.split(" +", lines[i+69])
            dpr[i] = float(word[-1].strip())

        index_u = ['pc[bar]', 'Tc[K]', 'Hc[KJ/kg]', 'Mc', 'gamma_c', 'SonicVelocity_c[m/s]']

        index_b = ['p[bar]', 'T[K]', 'rho[kg/m^3]', 'H[KJ/kg]', 'U[KJ/kg]', 'G[KJ/kg]', 'S[KJ/kg K]',\
                    'M', '(dLV/dLP)t', '(dLV/dLT)p', 'Cp[KJ/kg K]', 'gamma', 'SonicVelocity[m/s]']
                    
        index_d = ['p/pc', 'T/Tc', 'M/Mc', 'rho/rho_c', 'M_CJ', 'V_CJ[m/s]']

        self.unburned_gas = pd.Series(data=unb, index=index_u)
        self.burned_gas = pd.Series(data=bnd, index=index_b)
        self.detonation_parameters = pd.Series(data=dpr, index=index_d)

    def cea_run(self):
        """Run NASA-CEA"""
        cmd = subprocess.Popen(fcea2m, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        cmd.communicate((self.file_name + '\n').encode('utf-8'))

    def __init__(self, file_name, Tc, ER, pc,mixtures,AR):
        """Main part of this class"""
        self.file_name = file_name
        self.Tc = Tc
        self.ER = ER
        self.pc = pc
        self.mixtures=mixtures
        self.AR = AR
        os.chdir(cea_path)

    def PrintSeries(self):
#        print(self.unburned_gas)
#        print(self.burned_gas)
#        print(self.detonation_parameters)
        return self.unburned_gas,self.burned_gas,self.detonation_parameters


PP=df['初期圧力']
FAI=df['等量比']

pc=[]
Tc=[]
Hc=[]
Mc=[]
gammac=[]
SonicVelocityc=[]
p=[]
T=[]
rho=[]
H=[]
U=[]
G=[]
S=[]
M=[]
dLVdLP=[]
dLVdLT=[]
Cp=[]
gamma=[]
SonicVelocity=[]
ppc=[]
TTc=[]
MMc=[]
rhorhoc=[]
MCJ=[]
VCJ=[]


for i in range(len(a1)*n):
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15, FAI[i], PP[i],'C2H2,acetylene',0)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas


for i in range(len(a2)*n):
    i=len(a1)*n+i
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15,FAI[i], PP[i],'C2H4',0)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas

for i in range(len(a3)*n):
    i=len(a1)*n+len(a2)*n+i
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15, FAI[i], PP[i],'H2',0)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas


for i in range(len(a4)*n):
    i=len(a1)*n+len(a2)*n+len(a3)*n+i
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15, FAI[i], PP[i],'C2H6',0)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas


for i in range(len(a5)*n):
    i=len(a1)*n+len(a2)*n+len(a3)*n+len(a4)*n+i
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15, FAI[i], PP[i],'C2H2,acetylene',20)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas


for i in range(len(a6)*n):
    i=len(a1)*n+len(a2)*n+len(a3)*n+len(a4)*n+len(a5)*n+i
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15, FAI[i], PP[i],'C2H2,acetylene',30)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas




for i in range(len(a7)*n):
    i=len(a1)*n+len(a2)*n+len(a3)*n+len(a4)*n+len(a5)*n+len(a6)*n+i
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15, FAI[i], PP[i],'C2H4',40)
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        pc.append(unburned_gas[0])
        Tc.append(unburned_gas[1])
        Hc.append(unburned_gas[2])
        Mc.append(unburned_gas[3])
        gammac.append(unburned_gas[4])
        SonicVelocityc.append(unburned_gas[5])
        p.append(burned_gas[0])
        T.append(burned_gas[1])
        rho.append(burned_gas[2])
        H.append(burned_gas[3])
        U.append(burned_gas[4])
        G.append(burned_gas[5])
        S.append(burned_gas[6])
        M.append(burned_gas[7])
        dLVdLP.append(burned_gas[8])
        dLVdLT.append(burned_gas[9])
        Cp.append(burned_gas[10])
        gamma.append(burned_gas[11])
        SonicVelocity.append(burned_gas[12])
        ppc.append(detonation_parameters[0])
        TTc.append(detonation_parameters[1])
        MMc.append(detonation_parameters[2])
        rhorhoc.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        del calc
        del gas






df['pc[bar]']=pc
df['Tc[K]']=Tc
df['Hc[KJ/kg]']=Hc
df['Mc']=Mc
df['gamma_c']=gammac
df['SonicVelocity_c[m/s]']=gammac
df['p[bar]']=p
df['T[K]']=T
df['rho[kg/m^3]']=rho
df['H[KJ/kg]']=H
df['U[KJ/kg]']=U
df['G[KJ/kg]']=G
df['S[KJ/kg K]']=S
df['M']=M
df['(dLV/dLP)t']=dLVdLP
df['(dLV/dLT)p']=dLVdLT
df['Cp[KJ/kg K]']=Cp
df['gamma']=gamma
df['SonicVelocity[m/s]']=SonicVelocity
df['p/pc']=ppc
df['T/Tc']=TTc
df['M/Mc']=MMc
df['rho/rho_c']=rhorhoc
df['M_CJ']=MCJ
df['V_CJ[m/s]']=VCJ

print(df)
df.to_excel('/mnt/c/CEA/c2h2-c2h4-h2-c2h6-C2H2AR50-C2H2AR80-C2H4AR75.xlsx')