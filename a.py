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
import numpy as np
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import random


x1=[30,30,35,40,40,50,60,70,80,80,90,90,100]
y1=[60.531 ,52.180 ,36.280 ,30.208 ,28.536 ,23.924 ,20.417 ,16.366 ,13.264 ,14.769 ,14.965 ,14.306 ,11.234 ]

x2=[50,55,55,55,60,60,60,60,60,70,70,80,80,80,80,90,90,99,100]
y2=[41.19 ,36.11 ,39.47 ,31.12 ,35.00 ,32.50 ,36.08 ,33.40 ,32.54 ,29.93 ,25.23 ,20.83 ,21.39 ,22.78 ,24.54 ,23.06 ,24.30 ,19.27 ,19.16 ]


def exp_func_log(x, a, b):
    return a*np.log(x) + np.log(b)

def exp_func_log_fit(val1_quan, val2_quan):
    l_popt, l_pcov = curve_fit(exp_func_log, val1_quan, np.log(val2_quan), maxfev=10000, check_finite=False)
    return exp_func_log(val1_quan, *l_popt),l_popt

def log_to_exp(x,a,b):
    return np.exp(a*np.log(x) + np.log(b))



x=x1
y=y1
y_fit,l_popt=exp_func_log_fit(x,y)

ax=plt.subplot(1,1,1)
ax.plot(x,y,label='obs')
ax.plot(x,y_fit,label='model')
ax.set_xlabel('年次')
ax.set_ylabel('乳児死亡率')
plt.legend()
plt.show()
print('a : {},   b : {}'.format(l_popt[0],l_popt[1]))#求めたパラメータa,bを確認


def exp_func_log(x, b):
    return (-1)*np.log(x) + np.log(b)

def exp_func_log_fit(val1_quan, val2_quan):
    l_popt, l_pcov = curve_fit(exp_func_log, val1_quan, np.log(val2_quan), maxfev=10000, check_finite=False)
    return exp_func_log(val1_quan, *l_popt),l_popt

def log_to_exp(x,b):
    return np.exp((-1)*np.log(x) + np.log(b))



y_fit1,l_popt1=exp_func_log_fit(x2,y2)

y_fit1=log_to_exp(x2,l_popt1)


ax=plt.subplot(1,1,1)
ax.plot(x2,y2,label='obs')
ax.plot(x2,y_fit1,label='model')
ax.set_xlabel('年次')
ax.set_ylabel('乳児死亡率')
plt.legend()
plt.show()
print(' b : {}'.format(l_popt1) )
