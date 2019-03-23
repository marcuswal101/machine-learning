"""
Python 3.6

Program applies the new sabr model from Kennedy and FLock (2014) paper section 3 equation 24. 
calibration of alpha, rho, and nu is done using scipy.optimize of the objective function

Usage:
        Keep the input csv file in the same directory as this program and run it simply as: python new_sabr_model.py
        It will give the output in three csv sheets: outvol.csv,vol differences.csv, and parameters.csv

        These parameters.csv sheet contains the calibrated parameters

        the outvol.csv sheet has the sabr volatilies

Further Note: The program provides a pandas dataframe output of the csv file generated
"""

from __future__ import division
import xlrd
import math
import numpy as np
from scipy.optimize import minimize
import pandas as pd
 
#------Change the input_file and sheet name here

input_file = 'sample_format.xlsx'

input_sheet = 'tnotes'

beta_value = 0.5                  

dataframe_output_file = 'parameters.csv' #put the file name here to get the data's dataframe output: outvol.csv,vol differences.csv, or parameters.csv 

 
def SABR(alpha,beta,rho,nu,F,K,time,MKT): 
        """Implements the new sabr model"""
        if K <= 0:   
                VOL = 0
                diff = 0
        logFK = math.log(F/K)
        z = (nu / alpha*(1-beta))*(F**(1-beta)-K**(1-beta))
        x = (1/nu)*math.log( ( math.sqrt(1-2*rho*z+z**2) + z - rho ) / (1-rho) )
        gammaK = (K**beta - F**beta)/(K-F)
        yK = (K**(1-beta) - F**(1-beta))/(1-beta)
        yF = (F**(1-beta) - F**(1-beta))/(1-beta)
        DK = math.sqrt(alpha**2 + 2*alpha*rho*nu*yK + (nu**2)*(yK**2)*(K**beta))
        DF = math.sqrt(alpha**2 + 2*alpha*rho*nu*yF + (nu**2)*(yF**2)*(F**beta))
        gK = -(1/x**2)*math.log((logFK/x)*math.sqrt(F*K/(DF*DK)))
        VOL = (1/x)*logFK*(1 + (gK + 0.25*rho*nu*alpha*gammaK)*time)
        diff = VOL - MKT
        outvol.write('%r;' %round(VOL,4) )
        if MKT==0:
            diff = 0
            vol_diff.write('%s;' %'No market data')
        else:
            vol_diff.write('%r;' %round(diff,4) )
        
 
 
def smile(alpha,beta,rho,nu,F,K,time,MKT,i): 
    """F, time and the parameters are scalars, K and MKT are vectors, i is the index for tenor/expiry label"""
    """writes the implied volatilities found using the sabr calibration"""
    outvol.write('%s;%s;' %(label_ten[i],label_exp[i]))
    vol_diff.write('%s;%s;' %(label_ten[i],label_exp[i]))
    parameters.write('%s;%s;' %(label_ten[i],label_exp[i]))
 
    for j in range(len(K)):
        if K[0] <= 0:
            shift(F,K)
        SABR(alpha,beta,rho,nu,F,K[j],time,MKT[j])
 
    outvol.write('\n')
    vol_diff.write('\n')
    parameters.write('%f;%f;%f;%f;' %(alpha ,beta ,rho ,nu))
    parameters.write('\n')
 
 
def SABR_vol_matrix(alpha,beta,rho,nu,F,K,time,MKT): 
    """F, time and the parameters are vectors, K and MKT are matrices"""
    """writes the implied volatilities for different maturities and expiries"""
    outvol.write('%s;' %'SABR VOLATILITIES')
    outvol.write('\n')
    vol_diff.write('%s;' %'VOLATILITY DIFFERENCES')
    vol_diff.write('\n')
    parameters.write('%s;' %'PARAMETERS')
    parameters.write('\n')
    outvol.write('%s;%s;' %(' ','strikes:'))
    vol_diff.write('%s;%s;' %(' ','strikes:'))
    for j in range(len(strike_spreads)):
        outvol.write('%s;' %label_strikes[j])
        vol_diff.write('%s;' %label_strikes[j])
    outvol.write('\n')
    vol_diff.write('\n')
    parameters.write('%s;%s;%s;%s;%s;%s' %('tenor','expiry','alpha','beta','rho','nu'))
    parameters.write('\n')
 
    for i in range(len(F)):
        smile(alpha[i],beta[i],rho[i],nu[i],F[i],K[i],time[i],MKT[i],i)
 
 
def shift(F,K):
    """implements shifted sabr"""
    shift = 0.001 - K[0]
    for j in range(len(K)):
        K[j] = K[j] + shift
        F = F + shift   
 
 
 
def objfunc(par,F,K,time,MKT):
    """beta = par[1],rho = par[2],nu = par[3],alpha = par[0]"""
    """minimize the squared difference between market and implied volatilies"""
    sum_sq_diff = 0
    if K[0]<=0:
        shift(F,K)
    
    for j in range(len(K)):
                if MKT[j] == 0:   
                    diff = 0
                logFK = math.log(F/K[j])
                z = (par[3] / par[0]*(1-par[1]))*(F**(1-par[1])-K[j]**(1-par[1]))
                x = (1/par[3])*math.log( ( math.sqrt(1-2*par[2]*z+z**2) + z - par[2] ) / (1-par[2]) )
                gammaK = (K[j]**par[1] - F**par[1])/(K[j]-F)
                yK = (K[j]**(1-par[1]) - F**(1-par[1]))/(1-par[1])
                yF = (F**(1-par[1]) - F**(1-par[1]))/(1-par[1])
                DK = math.sqrt(par[0]**2 + 2*par[0]*par[2]*par[3]*yK + (par[3]**2)*(yK**2)*(K[j]**par[1]))
                DF = math.sqrt(par[0]**2 + 2*par[0]*par[2]*par[3]*yF + (par[3]**2)*(yF**2)*(F**par[1]))
                gK = -(1/x**2)*math.log((logFK/x)*math.sqrt(F*K[j]/(DF*DK)))
                VOL = (1/x)*logFK*(1 + (gK + 0.25*par[2]*par[3]*par[0]*gammaK)*time)
                diff = VOL - MKT[j]         
                sum_sq_diff = sum_sq_diff + diff**2
                obj = math.sqrt(sum_sq_diff)
                return obj


def calibration(starting_par,F,K,time,MKT):
        """optimizes the objfunc to get the parameters satisfying the problem"""
        for i in range(len(F)):
               x0 = starting_par
               bnds = ( (0.001,None) , (0,1) , (-0.999,0.999) , (0.001,None)  )
               res = minimize(objfunc, x0 , (F[i],K[i],time[i],MKT[i]) ,bounds = bnds, method='SLSQP')
               alpha[i] = res.x[0]
               #beta[i] = res.x[1]          #uncomment this for beta calibration
               beta[i] = beta_value                                
               rho[i] = res.x[2]
               nu[i] = res.x[3]
       
 
 
######## inputs and outputs #########################################
 
outvol = open('outvol.csv', 'w')             # file output of volatilities
vol_diff = open('vol differences.csv', 'w')  # file output differences between SABR and Market volatilities
parameters = open('parameters.csv', 'w')     # file output parameters
 
while True:
        try:
                file_input = xlrd.open_workbook(input_file)
        except:
                print ('Input file is not in the directory!')
        break
  
Market_data = file_input.sheet_by_name(input_sheet)
 
######## set data characteristics ###############################
      
strike_spreads=[]
j=0
while True:
    try:
        strike_spreads.append(int(Market_data.cell(1,3+j).value))
        j = j+1
    except:
        break
num_strikes = len(strike_spreads)
 
expiries=[]
i=0
while True:
        try:
            expiries.append(Market_data.cell(2+i,1).value)
            i = i + 1
        except:
            break
 
tenors=[]
i=0
while True:
    try:
        tenors.append(Market_data.cell(2+i,0).value)
        i = i + 1
    except:
        break
 
 
# to create the ATM forward rates
F = []
i=0
while True:
    try:
        F.append(Market_data.cell(2+i,2).value)
        i = i+1
    except:
        break
 
# to create the strike grid
K = np.zeros((len(F),num_strikes))
for i in range(len(F)):
    for j in range(num_strikes):
        K[i][j] = F[i] + 0.0001*(strike_spreads[j])  
 
# to create market volatilities            
MKT = np.zeros((len(F),num_strikes))
for i in range(len(F)):
    for j in range(num_strikes):
        MKT[i][j] = Market_data.cell(2+i,3+j).value
 
 
# set starting parameters - these will be used to begin the calibration 
starting_guess = np.array([0.001,0.5,0,0.001])
alpha = len(F)*[starting_guess[0]]
beta = len(F)*[starting_guess[1]]
rho = len(F)*[starting_guess[2]]
nu = len(F)*[starting_guess[3]]
 
 
######## set labels ###################################################
 
exp_dates = len(expiries)*[0]
for i in range(len(expiries)):
     exp_dates[i] = str((round(expiries[i],3)))

 
ten_dates = len(tenors)*[0]
for i in range(len(tenors)):
    if tenors[i] < 1:
        ten_dates[i] = str(int(round(12*tenors[i])))+'m'
    else:
        ten_dates[i] = str(int(round(tenors[i])))+'y'
        if tenors[i]-round(tenors[i]) > 0:
            ten_dates[i] = ten_dates[i]+str(int(round((12*(round(tenors[i],2)-int(tenors[i]))))))+'m'
        elif tenors[i]-round(tenors[i]) < 0:
            ten_dates[i] = str(int(round(tenors[i]))-1)+'y'
            ten_dates[i] = ten_dates[i]+str(int(round((12*(round(tenors[i],2)-int(tenors[i]))))))+'m'
 
label_exp = exp_dates
label_ten = ten_dates
label_strikes = num_strikes*[0]
for i in range(num_strikes):
    if strike_spreads[i] == 0 :
        label_strikes[i] = 0
    else:
        label_strikes[i] = str(strike_spreads[i])
 
 
######## Call the functions #################################
 
calibration(starting_guess,F,K,expiries,MKT)
 
SABR_vol_matrix(alpha,beta,rho,nu,F,K,expiries,MKT)

######## Close output files #################################
 
outvol.close()
vol_diff.close()
parameters.close() 

#------Get pandas datafram output
df_param = pd.read_csv(dataframe_output_file,engine="python",index_col=False, header=1, delimiter='\;')
print (df_param)





