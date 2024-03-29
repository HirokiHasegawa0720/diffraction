import pandas as pd
import numpy as np
import random
import os
import subprocess
import re

df = pd.read_excel('/mnt/c/CEA/exp_database_20210629.xlsx')

cea_path = "/mnt/c/CEA/cea-exec"    # PATH to NASA-CEA
fcea2m = "/mnt/c/CEA/cea-exec/FCEA2m.exe" # fcea2m.exe

df = df[( df['propagation'] == 'yes')]
df = df.replace({'fuel': {'C2H2': 'C2H2,acetylene'}})
df = df.replace({'fuel': {'C3H6': 'C3H6,propylene'}})

Equivalentratio = list(df['phi'])
Fuel = list(df['fuel'])
Coefficientfuel = list(df['fuel st. coeff.'])
Oxidizer = list(df['oxidizer'])
CoefficientOxidizer = list(df['ox st. coeff.'])
Diluent = list(df['diluent'])
CoefficientDiluent = list(df['diluent st. coeff.'])
T0 = list(df['T0'])
P0 = list(df['p0'])
Lr = list(df['lr'])
Lc = list(df['lc'])

def calc_double(n):
    return n/100

P0 = list(map(calc_double,P0))

d = {'P0':P0}

dfs = pd.DataFrame(d)

dfs['P0'] = P0
dfs['T0'] = T0
dfs['Lr'] = Lr
dfs['Lc'] = Lc
dfs['Equivalentratio'] = Equivalentratio
dfs['Fuel'] = Fuel
dfs['Coefficientfuel'] = Coefficientfuel
dfs['Oxidizer'] = Oxidizer
dfs['CoefficientOxidizer'] = CoefficientOxidizer
dfs['Diluent'] = Diluent
dfs['CoefficientDiluent'] = CoefficientDiluent

class NASAcea():

    def det_inp(self):
        """Create .inp file (detonation)"""
        if self.Diluentc is np.nan:
            print('nai')
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
                        "end\n" )  % (self.Tc , self.Equivalentratioc , self.Pc , self.Fuelc , self.Tc , self.Oxidizerc , self.Tc )
        
        else:
            print('ari')
            a = self.Coefficientfuelc / ( self.Coefficientfuelc + self.CoefficientOxidizerc + self.CoefficientDiluentc ) * 100
            b = self.CoefficientOxidizerc / ( self.Coefficientfuelc + self.CoefficientOxidizerc + self.CoefficientDiluentc ) * 100
            c = self.CoefficientDiluentc / ( self.Coefficientfuelc + self.CoefficientOxidizerc + self.CoefficientDiluentc ) * 100      
            setting = ( "problem\n"
                        "    detonation\n"
                        "    t = %.2f\n\n\n\n\n\n"
                        "    r,e = %.2f\n"
                        "    p,bar = %.3f\n"
                        "reactant\n"
                        "    fuel = %s  mole = %.3f\n"
                        "    fuel = %s  mole = %.3f\n"
                        "    oxid = %s  mole = %.3f\n"
                        "outp\n"
                        "    short\n"
                        "end\n" )  % (self.Tc , self.Equivalentratioc , self.Pc , self.Fuelc , a , self.Diluentc , c  , self.Oxidizerc , b )
                        
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

        index_u = ['p0[bar]', 'T0[K]', 'H0[KJ/kg]', 'M0kg/kmol', 'gamma_0', 'SonicVelocity_0[m/s]']

        index_b = ['p_CJ[bar]', 'T_CJ[K]', 'rho_CJ[kg/m^3]', 'H_CJ[KJ/kg]', 'U_CJ[KJ/kg]', 'G_CJ[KJ/kg]', 'S_CJ[KJ/kg K]',\
                    'M_CJkg/kmol', '(dLV/dLP)t_CJ', '(dLV/dLT)p_CJ', 'Cp_CJ[KJ/kg K]', 'gamma_CJ', 'SonicVelocity_CJ[m/s]']
                    
        index_d = ['p_CJ/p0', 'T_CJ/T0', 'M_CJ/M0', 'rho_CJ/rho_0', 'M_CJ', 'V_CJ[m/s]']

        self.unburned_gas = pd.Series(data=unb, index=index_u)
        self.burned_gas = pd.Series(data=bnd, index=index_b)
        self.detonation_parameters = pd.Series(data=dpr, index=index_d)

    def cea_run(self):
        """Run NASA-CEA"""
        cmd = subprocess.Popen(fcea2m, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        cmd.communicate((self.file_name + '\n').encode('utf-8'))

    def __init__(self, file_name, Pc , Tc , Equivalentratioc ,Fuelc , Coefficientfuelc , Oxidizerc  , CoefficientOxidizerc , Diluentc , CoefficientDiluentc):
        """Main part of this class"""
        self.file_name = file_name
        self.Pc = Pc
        self.Tc = Tc
        self.Equivalentratioc = Equivalentratioc
        self.Fuelc = Fuelc
        self.Coefficientfuelc = Coefficientfuelc
        self.Oxidizerc = Oxidizerc
        self.CoefficientOxidizerc = CoefficientOxidizerc
        self.Diluentc = Diluentc
        self.CoefficientDiluentc = CoefficientDiluentc
        os.chdir(cea_path)

    def PrintSeries(self):
        return self.unburned_gas,self.burned_gas,self.detonation_parameters
    
class NASAceashock():

    def det_inp(self):
        """Create .inp file (shock)"""
        a = self.Coefficientfuelv / ( self.Coefficientfuelv + self.CoefficientOxidizerv + self.CoefficientDiluentv ) * 100
        b = self.CoefficientOxidizerv / ( self.Coefficientfuelv + self.CoefficientOxidizerv + self.CoefficientDiluentv ) * 100
        c = self.CoefficientDiluentv / ( self.Coefficientfuelv + self.CoefficientOxidizerv + self.CoefficientDiluentv ) * 100

        d = self.Coefficientfuelv / ( self.Coefficientfuelv + self.CoefficientOxidizerv ) * 100
        e = self.CoefficientOxidizerv / ( self.Coefficientfuelv + self.CoefficientOxidizerv ) * 100

        if self.Diluentv is np.nan:
            print('nai')
            setting = ( "problem\n"
                        "    shock\n"
                        "    t = %.2f\n"
                        "    p,bar = %.3f\n\n"
                        "    mach1 = %.3f\n"
                        "reactant\n"
                        "    name = %s  mole = %.2f\n"
                        "    name = %s  mole = %.2f\n"
                        "outp\n"
                        "    short\n"
                        "end\n" )  % (self.Tv , self.Pv , self.M_CJ , self.Fuelv , d , self.Oxidizerv , e )

        else:
            print('ari')   
            setting = ( "problem\n"
                        "    shock\n"
                        "    t = %.2f\n"
                        "    p,bar = %.3f\n"
                        "    mach1 = %.3f\n"
                        "reactant\n"
                        "    name = %s  mole = %.3f\n"
                        "    name = %s  mole = %.3f\n"
                        "    name = %s  mole = %.3f\n"
                        "outp"
                        "    short\n"
                        "end\n" )  % (self.Tv , self.Pv , self.M_CJ , self.Fuelv , a , self.Diluentv , c , self.Oxidizerv , b)
                        
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

        ini = np.zeros(7)
        sho = np.zeros(12)
        dpr = np.zeros(5)

        for i in range(1):
            word = re.split(" +", lines[i+43])
            ini[i] = float(word[-1].strip())
        word = re.split(" +", lines[46])
        if len(word) == 6:
            ini[1] = float(word[-2]) * 10**float(word[-1].strip())
        else:
            num = word[-1].split('-')
            ini[1] = float(num[0]) * 10**float(num[1].strip())

        for i in range(3):
            word = re.split(" +", lines[i+48])
            ini[i+2] = float(word[-1].strip())
        word = re.split(" +", lines[53])
        ini[5] = float(word[-1].strip())
        word = re.split(" +", lines[55])
        ini[6] = float(word[-1].strip())
        for i in range(3):
            word = re.split(" +", lines[i+58])
            sho[i] = float(word[-1].strip())
        word = re.split(" +", lines[61])
        if len(word) == 6:
            sho[3] = float(word[-2]) * 10**float(word[-1].strip())
        else:
            num = word[-1].split('-')
            sho[3] = float(num[0]) * 10**float(num[1].strip())
        for i in range(4):
            word = re.split(" +", lines[i+62])
            sho[i+4] = float(word[-1].strip())
        for i in range(4):
            word = re.split(" +", lines[i+67])
            sho[i+8] = float(word[-1].strip())

        for i in range(5):
            word = re.split(" +", lines[i+72])
            dpr[i] = float(word[-1].strip())

        index_u = ['U0[m/s]', 'rho0[kg/m^3]', 'U0[kg/kj]', 'G0[kg/kj]', 'S0[kg/kj]', 'Cp0[kj/kgK]','SonicVelocity0[m/s]']
        index_b = ['Uvn[m/s]','Pvn[bar]', 'Tvn[K]', 'rhovn[kg/m^3]', 'Hvn[KJ/kg]', 'Uvn[KJ/kg]', 'Gvn[KJ/kg]', 'Svn[KJ/kg K]',\
                    'Mvn[kg/kmol]', 'Cpvn[KJ/kg K]', 'gammavn', 'SonicVelocityvn[m/s]']
                    
        index_d = ['Pvn/P0', 'Tvn/Tc', 'Mvn[kg/kmol]/M0[kg/kmol]', 'rhovn/rho0', 'V_vn[m/s]']

        self.unburned_gas = pd.Series(data=ini, index=index_u)
        self.shock_gas = pd.Series(data=sho, index=index_b)
        self.shock_parameters = pd.Series(data=dpr, index=index_d)

    def cea_run(self):
        """Run NASA-CEA"""
        cmd = subprocess.Popen(fcea2m, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        cmd.communicate((self.file_name + '\n').encode('utf-8'))

    def __init__(self, file_name, Pv , Tv , Equivalentratiov ,Fuelv , Coefficientfuelv , Oxidizerv  , CoefficientOxidizerv , Diluentv , CoefficientDiluentv , M_CJ ):
        """Main part of this class"""
        self.file_name = file_name
        self.Pv = Pv
        self.Tv = Tv
        self.Equivalentratiov = Equivalentratiov
        self.Fuelv = Fuelv
        self.Coefficientfuelv = Coefficientfuelv
        self.Oxidizerv = Oxidizerv
        self.CoefficientOxidizerv = CoefficientOxidizerv
        self.Diluentv = Diluentv
        self.CoefficientDiluentv = CoefficientDiluentv
        self.M_CJ = M_CJ
        os.chdir(cea_path)

    def PrintSeries(self):
        return self.unburned_gas,self.shock_gas,self.shock_parameters


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

for i in range(len(P0)):
    if __name__ == "__main__":
        print(i)
        calc = NASAcea("test", P0[i] , T0[i] , Equivalentratio[i] ,Fuel[i] , Coefficientfuel[i] , Oxidizer[i]  , CoefficientOxidizer[i] , Diluent[i] , CoefficientDiluent[i] )
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        burned_gas=gas[1]
        detonation_parameters=gas[2]
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
        del calc
        del gas

              
U0ms=[]
rho0=[]
U0kgkj=[]
G0=[]
S0=[]
Cp0=[]
SonicVelocity0=[]
Uvnms=[]
Pvn=[]
Tvn=[]
rhovn=[]
Hvn=[]
Uvnkgkj=[]
Gvn=[]
Svn=[]
Mvnkgkmol=[]
Cpvn=[]
gammavn=[]
SonicVelocityvn=[]
PvnP0=[]
TvnTc=[]
MvnM0=[]
rhovnrho0=[]
V_vn=[]


for i in range(len(P0)):
    if __name__ == "__main__":
        print(i)
        calc = NASAceashock("test", P0[i] , T0[i] , Equivalentratio[i] ,Fuel[i] , Coefficientfuel[i] , Oxidizer[i]  , CoefficientOxidizer[i] , Diluent[i] , CoefficientDiluent[i] , MCJ[i])
        calc.det_inp()
        calc.cea_run()
        calc.det_out()
        gas=calc.PrintSeries()
        unburned_gas=gas[0]
        shock_gas=gas[1]
        shock_parameters=gas[2]
        U0ms.append(unburned_gas[0])
        rho0.append(unburned_gas[1])
        U0kgkj.append(unburned_gas[2])
        G0.append(unburned_gas[3])
        S0.append(unburned_gas[4])
        Cp0.append(unburned_gas[5])
        SonicVelocity0.append(unburned_gas[6])
        Uvnms.append(shock_gas[0])
        Pvn.append(shock_gas[1])
        Tvn.append(shock_gas[2])
        rhovn.append(shock_gas[3])
        Hvn.append(shock_gas[4])
        Uvnkgkj.append(shock_gas[5])
        Gvn.append(shock_gas[6])
        Svn.append(shock_gas[7])
        Mvnkgkmol.append(shock_gas[8])
        Cpvn.append(shock_gas[9])
        gammavn.append(shock_gas[10])
        SonicVelocityvn.append(shock_gas[11])
        PvnP0.append(shock_parameters[0])
        TvnTc.append(shock_parameters[1])
        MvnM0.append(shock_parameters[2])
        rhovnrho0.append(shock_parameters[3])
        V_vn.append(shock_parameters[4])
        del calc
        del gas


dfs['H0[KJ/kg]']=H0
dfs['M0[kg/kmol]']=M0kgkmol
dfs['γ0[-]']=gamma0
dfs['a0[m/s]']=SonicVelocity0
dfs['U0[m/s]']=U0ms
dfs['rho0[kg/m^3]']=rho0
dfs['U0[kg/kj]']=U0kgkj
dfs['G0[kg/kj]']=G0
dfs['S0[kg/kj]']=S0
dfs['Cp0[kj/kgK]']=Cp0

dfs['Uvn[m/s]']=Uvnms
dfs['Pvn[bar]']=Pvn
dfs['Tvn[K]']=Tvn
dfs['rhovn[kg/m^3]']=rhovn
dfs['Hvn[KJ/kg]']=Hvn
dfs['Uvn[KJ/kg]']=Uvnkgkj
dfs['Gvn[KJ/kg]']=Gvn
dfs['Svn[KJ/kg K]']=Svn
dfs['Mvn[kg/kmol]']=Mvnkgkmol
dfs['Cpvn[KJ/kg K]']=Cpvn
dfs['γvn[-]']=gammavn
dfs['avn[m/s]']=SonicVelocityvn
dfs['Pvn/P0']=PvnP0
dfs['Tvn/Tc']=TvnTc
dfs['Mvn[kg/kmol]/M0[kg/kmol]']=MvnM0
dfs['rhovn/rho0']=rhovnrho0
dfs['V_vn[m/s]']=V_vn

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

dfs.to_excel('/mnt/c/CEA/all_mixtureexp.xlsx')

