import pandas as pd
import numpy as np
import random
import os
import subprocess
import re

df = pd.read_excel('/mnt/c/CEA/DetonationDatabase.xlsx')

cea_path = "/mnt/c/CEA/cea-exec"    # PATH to NASA-CEA
fcea2m = "/mnt/c/CEA/cea-exec/FCEA2m.exe" # fcea2m.exe

class Predata:

    def __init__(self,n):
        self.n = n
    
    def lr(self,P,a):
        return (a)*P**(-1)
    
    def many(self,params):
        a=[]
        for i in range(len(params)):
            for j in range(self.n):
                a.append(params[i])
        return a
    
    def manyPLR(self,Coefficienta):
        P=[]
        LR=[]
        for i in range(len(Coefficienta)):
            randomp = random.uniform(40,100)
            P.append(randomp)
        for i in range(len(Coefficienta)):
            LR.append(self.lr(P[i],Coefficienta[i]))
        return P,LR
#randomp = random.uniform(10,100)
n=10
'''
P=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]*40
 for i in range(len(Coefficienta)):
            randomp = random.uniform(10,100)
            P.append(randomp)
'''
p1 = Predata(n)
Coefficienta = p1.many(df['Coefficienta'])
Equivalentratio = p1.many(df['Equivalentratio'])
Fuel = p1.many(df['Fuel'])
Coefficientfuel = p1.many(df['Coefficientfuel'])
Oxidant = p1.many(df['Oxidant'])
CoefficientOxidant = p1.many(df['CoefficientOxidant'])
Diluent = p1.many(df['Diluent'])
CoefficientDiluent = p1.many(df['CoefficientDiluent'])
Diluentratio = p1.many(df['Diluentratio'])
Fuelratio1 = p1.many(df['Fuelratio1'])
Fuelratio2 = p1.many(df['Fuelratio2'])
fuel_ratio1 = p1.many(df['fuel_ratio1'])
oxidant_ratio2 = p1.many(df['oxidant_ratio2'])
diluent_ratio3 = p1.many(df['diluent_ratio3'])

PLR = p1.manyPLR(Coefficienta)

print(len(Coefficienta))

P = PLR[0]
LR = PLR[1]
def calc_double(n):
    return n/100
P =list(map(calc_double,P))

d = {'Coefficienta':Coefficienta}

dfs = pd.DataFrame(d)

dfs['Coefficienta'] = Coefficienta
dfs['Equivalentratio'] = Equivalentratio
dfs['Fuel'] = Fuel
dfs['Coefficientfuel'] = Coefficientfuel
dfs['Oxidant'] = Oxidant
dfs['CoefficientOxidant'] = CoefficientOxidant
dfs['Diluent'] = Diluent
dfs['CoefficientDiluent'] = CoefficientDiluent
dfs['Diluentratio'] = Diluentratio
dfs['Fuelratio1'] = Fuelratio1
dfs['Fuelratio2'] = Fuelratio2
dfs['fuel_ratio1'] = fuel_ratio1
dfs['oxidant_ratio2'] = oxidant_ratio2
dfs['diluent_ratio3'] = diluent_ratio3

dfs['P'] = P
dfs['LR'] = LR


class NASAcea():

    def det_inp(self):
        """Create .inp file (detonation)"""

        fuel = self.Fuel
        oxidizer = self.Oxidant
        if self.Diluent == 0:
            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n\n"
                        "reactant\n"
                        "    fuel = %s  t(k) = %.2f\n"
                        "    oxid = %s   t(k) = %.2f\n"
                        "output short\n"
                        "output transport\n"
                        "end\n" )  % (self.Tc , self.Equivalentratio , self.P , fuel , self.Tc , oxidizer , self.Tc )
        
        if self.Diluent != 0:
            diluent = self.Diluent           
            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n\n\n\n\n\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n"
                        "reactant\n"
                        "    fuel = %s  mole = %.3f\n"
                        "    fuel = %s  mole = %.3f\n"
                        "    oxid = %s  mole = %.3f\n"
                        "output short\n"
                        "output transport\n"
                        "end\n" )  % (self.Tc , self.Equivalentratio , self.P , fuel, self.Fuelratio1  , diluent , self.Fuelratio2  , oxidizer , 100 )
                        
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
        tran= np.zeros(3)
        dpr = np.zeros(6)

        for i in range(6):
            word = re.split(" +", lines[i+44])
            unb[i] = float(word[-1].strip())

        for i in range(2):
            word = re.split(" +", lines[i+53])
            bnd[i] = float(word[-1].strip())
        word = re.split(" +", lines[55])
        if len(word) == 6:
            bnd[2] = float(word[-2]) * 10**float(word[-1].strip())
        else:
            num = word[-1].split('-')
            bnd[2] = float(num[0]) * 10**float(num[1].strip())
        for i in range(4):
            word = re.split(" +", lines[i+56])
            bnd[i+3] = float(word[-1].strip())
        for i in range(6):
            word = re.split(" +", lines[i+61])
            
            bnd[i+7] = float(word[-1].strip())
        
        word = re.split(" +", lines[71])
        tran[0] = float(word[-1].strip())
        word = re.split(" +", lines[76])
        tran[1] = float(word[-1].strip())
        word = re.split(" +", lines[77])
        tran[2] = float(word[-1].strip())

        for i in range(6):
            word = re.split(" +", lines[i+87])
            dpr[i] = float(word[-1].strip())

        index_u = ['p0[bar]', 'T0[K]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0', 'SonicVelocity_0[m/s]']

        index_b = ['p_CJ[bar]', 'T_CJ[K]', 'rho_CJ[kg/m^3]', 'H_CJ[KJ/kg]', 'U_CJ[KJ/kg]', 'G_CJ[KJ/kg]', 'S_CJ[KJ/kg K]',\
                    'M_CJkg/kmol', '(dLV/dLP)t_CJ', '(dLV/dLT)p_CJ', 'Cp_CJ[KJ/kg K]', 'gamma_CJ', 'SonicVelocity_CJ[m/s]']
                    
        index_d = ['p_CJ/p0', 'T_CJ/T0', 'M_CJ/M0', 'rho_CJ/rho_0', 'M_CJ', 'V_CJ[m/s]']

        index_t = ['VISC,MILLIPOISE', 'CONDUCTIVITY', 'PRANDTL NUMBER']

        self.unburned_gas = pd.Series(data=unb, index=index_u)
        self.burned_gas = pd.Series(data=bnd, index=index_b)
        self.detonation_parameters = pd.Series(data=dpr, index=index_d)
        self.transport = pd.Series(data=tran, index=index_t)

    def cea_run(self):
        """Run NASA-CEA"""
        cmd = subprocess.Popen(fcea2m, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        cmd.communicate((self.file_name + '\n').encode('utf-8'))

    def __init__(self, file_name, Tc, Equivalentratio , Fuel , Oxidant , Diluent , Fuelratio1 , Fuelratio2 , P ):
        """Main part of this class"""
        self.file_name = file_name
        self.Tc = Tc
        self.Equivalentratio = Equivalentratio
        self.Fuel = Fuel
        self.Oxidant = Oxidant
        self.Diluent = Diluent
        self.Fuelratio1 = Fuelratio1
        self.Fuelratio2 = Fuelratio2
        self.P = P
        os.chdir(cea_path)

    def PrintSeries(self):
        return self.unburned_gas,self.burned_gas,self.detonation_parameters,self.transport

    
p0=[]
T0=[]
H0=[]
M0kgkmol=[]
gamma0=[]
SonicVelocity0=[]
pcj=[]
Tcj=[]
rhocj=[]
Hcj=[]
Ucj=[]
Gcj=[]
Scj=[]
Mcjkgkmol=[]
dLVdLPcj=[]
dLVdLTcj=[]
Cpcj=[]
gammacj=[]
SonicVelocitycj=[]
ppccj=[]
TTccj=[]
MMccj=[]
rhorhoccj=[]
MCJ=[]
VCJ=[]
VISCMILLIPOISE=[]
CONDUCTIVITY=[]
PRANDTLNUMBER=[]

for i in range(len(Coefficienta)):
    if __name__ == "__main__":
        calc = NASAcea("test", 298.15 , Equivalentratio[i] , Fuel[i] , Oxidant[i] , Diluent[i]  , Fuelratio1[i] , Fuelratio2[i] , P[i] )
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
        trans_parameters=gas[3]
        p0.append(unburned_gas[0])
        T0.append(unburned_gas[1])
        H0.append(unburned_gas[2])
        M0kgkmol.append(unburned_gas[3])
        gamma0.append(unburned_gas[4])
        SonicVelocity0.append(unburned_gas[5])
        pcj.append(burned_gas[0])
        Tcj.append(burned_gas[1])
        rhocj.append(burned_gas[2])
        Hcj.append(burned_gas[3])
        Ucj.append(burned_gas[4])
        Gcj.append(burned_gas[5])
        Scj.append(burned_gas[6])
        Mcjkgkmol.append(burned_gas[7])
        dLVdLPcj.append(burned_gas[8])
        dLVdLTcj.append(burned_gas[9])
        Cpcj.append(burned_gas[10])
        gammacj.append(burned_gas[11])
        SonicVelocitycj.append(burned_gas[12])
        ppccj.append(detonation_parameters[0])
        TTccj.append(detonation_parameters[1])
        MMccj.append(detonation_parameters[2])
        rhorhoccj.append(detonation_parameters[3])
        MCJ.append(detonation_parameters[4])
        VCJ.append(detonation_parameters[5])
        VISCMILLIPOISE.append(trans_parameters[0])
        CONDUCTIVITY.append(trans_parameters[1])
        PRANDTLNUMBER.append(trans_parameters[2])
        del calc
        del gas

 





dfs['p0[bar]']=p0
dfs['T0[K]']=T0
dfs['H0[KJ/kg]']=H0
dfs['M0[kg/kmol]']=M0kgkmol
dfs['γ0[-]']=gamma0
dfs['a0[m/s]']=SonicVelocity0

dfs['pcj[bar]']=pcj
dfs['Tcj[K]']=Tcj
dfs['rhocj[kg/m^3]']=rhocj
dfs['Hcj[KJ/kg]']=Hcj
dfs['Ucj[KJ/kg]']=Ucj
dfs['Gcj[KJ/kg]']=Gcj
dfs['Scj[KJ/kg K]']=Scj
dfs['Mcj[kg/kmol]']=Mcjkgkmol
dfs['(dLV/dLP)tcj']=dLVdLPcj
dfs['(dLV/dLT)pcj']=dLVdLTcj
dfs['Cpcj[KJ/kg K]']=Cpcj
dfs['γcj[-]']=gammacj
dfs['acj[m/s]']=SonicVelocitycj
dfs['p_CJ/p0']=ppccj
dfs['T_CJ/T0']=TTccj
dfs['M_CJ/M0']=MMccj
dfs['rho_CJ/rho0']=rhorhoccj
dfs['Mcj[-]']=MCJ
dfs['Vcj[m/s]']=VCJ
dfs['VISCMILLIPOISECJ']=VISCMILLIPOISE
dfs['CONDUCTIVITYCJ']=CONDUCTIVITY
dfs['PRANDTLNUMBERCJ']=PRANDTLNUMBER


dfs.to_excel('/mnt/c/CEA/all_mixturetran.xlsx')
