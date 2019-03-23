import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import pylab as pl
from datetime import datetime

################features functions - step 2############
def get_width_and_mid(df,k):
    '''
    Returns width of best market and midpoint for each data point in DataFrame
    of book data
    '''
    result = []
    best_bid = df['Bidprice'].rolling(k).apply(lambda x: x[0], raw = True)	
    best_ask = df['Askprice'].rolling(k).apply(lambda x: x[len(x) - 1], raw = True) 
    return(best_ask-best_bid)

#########import data from csv

k = 20 #delay
p = 5  #lags
activate_multithres = False
threshold = 0.006 #btc/usd and eth/usd
#threshold = 0.000006
strategy = 'A' # or 'B'. However, A is recommeded for cryptocurrencies 
trade_amount = 1 # 1 BTC or ETH etc
trade_fees = 0
# volatility_lvl = 0.55

# inpath = '/home/marcus/pyalgo/btcusd_lob_date.csv' # 1 day data
# inpath = '/home/marcus/pyalgo/ETHUSD_lob_date.csv' # 1 day data

# folder = '/ADAETH/' # 7 days data
# folder = '/EOSETH/' # 7 days data
# folder = '/NEOETH/' # 7 days data
# folder = '/BTCUSDT/' # 7 days data
# folder = '/ETHUSDT/' # 7 days data
folder = '/LTCETH/'
# inpath = '/home/marcus/pyalgo/lob_jsons' + folder + 'concatenated.csv' 


def trade_algo(inpath, threshold):

	df = pd.read_csv(inpath)
	n = len(df.index)   #data count  - test 2677
	if 'InstrumentID' in df:
		df = df.drop(['InstrumentID'],axis =1)
	else:
		pass
	#--make new column volume order imbalance
	df['VOI'] = df['Bidvolume'] - df['Askvolume']
	#--add spread, OIR, and midprice
	df['spread'] = df['Askprice'] - df['Bidprice']
	df['OIR'] = (df['Bidvolume'] - df['Askvolume'])/(df['Bidvolume'] + df['Askvolume'])
	df['midprice'] = (df['Bidprice'] + df['Askprice'])/2
	#include volality based on midprice
	df['Log_Ret'] = np.log(df['midprice'] / df['midprice'].shift(1))
	df['Volatility'] = df['Log_Ret'].rolling(252).std() * np.sqrt(252)
	#--find price midresponse
	roll_avg = df['midprice'].rolling(k).mean() #rolling mean of midprice
	mid_response = roll_avg - df['midprice'].shift(k - 1) 
	df['rollavg'] = roll_avg
	df['midpshift'] = df['midprice'].shift(k - 1)
	df['midresponse'] = mid_response
	#features column
	df['width'] = get_width_and_mid(df,k)
	if strategy == 'A':
		#--find jth lag VOIs
		for lag in range(p):
			voi_lag = df['VOI'].shift(lag + 1)
			df['VOI_' + str(lag + 1)] = voi_lag 
		#--drop k - 1 rows so there are no NaN values due to rolling done earlier
		df = df.iloc[k-1:]
		#--regression strategy A: midresponse (y) and VOI(s) (x) and width and aggressor features
		result = sm.ols(formula="midresponse ~ VOI + VOI_1 + VOI_2 + VOI_3 + VOI_4 + VOI_5 + width", data=df).fit()
		summary = result.summary()
		midr_fit = result.fittedvalues #dataframe of the fitted value of midresponse
		midr_avg = result.fittedvalues.mean() #average of the predicted mid response value. Use it to set threshold
		# print('Average MidResponse is:',midr_avg)
	#--regression strategy B: dMid.Response(y) , VOI=VOI/spread (x1), OIR=OIR/spread) (x2)
	elif strategy == 'B':
		for lag in range(p): #finding VOI/Spread variables
			spread_lag = df['spread'].shift(lag + 1) 
			voi_lag = df['VOI'].shift(lag + 1)
			df['spVOI_' + str(lag + 1)] = voi_lag / spread_lag

		for lag in range(p): #finding OIR/Spread variables
			spread_lag = df['spread'].shift(lag + 1) 
			oir_lag = df['OIR'].shift(lag + 1)
			df['spOIR_' + str(lag + 1)] = oir_lag / spread_lag
		df = df.iloc[k-1:]
		result = sm.ols(formula="midresponse ~ spVOI_1 + spVOI_2 + spVOI_3 + spVOI_4 + spVOI_5 + spOIR_1 + spOIR_2 + spOIR_3 + spOIR_4 + spOIR_5", data=df).fit()
		summary = result.summary()
		midr_fit = result.fittedvalues #dataframe of the fitted value of midresponse
		midr_avg = result.fittedvalues.mean() #average of the predicted mid response value. Use it to set threshold
		# print('Average MidResponse is:',midr_avg)
	else:
		pass
	#--running trade simulation
	df['midr_fit'] = midr_fit #add the fitted y values to the dataframe
	df = df.reset_index() #reset index values to start from zero

	buyprices = []
	sellprices = []
	buytime = []
	selltime = []
	for index, row in df.iterrows(): #loop over the dataframes
		# if row["Volatility"] >= volatility_lvl:
			if row["midr_fit"] <= -1*threshold:
				buyprice = row["Askprice"]
				buy_time = row["Time"]
				buyprices.append(buyprice)
				buytime.append(buy_time)
			elif row["midr_fit"] < threshold:
				sellprice = row["Bidprice"]
				sell_time = row["Time"]
				sellprices.append(sellprice)
				selltime.append(sell_time)
			else:
				pass

		# else:
		# 	pass

	pnl = []
	time_format = "%H:%M:%S"
	for trade in zip(selltime,sellprices,buytime,buyprices):
		print("Time:%s Short at Price:%s Time:%s Long at Price:%s"%(trade[0],trade[1],trade[2],trade[3]))
		print("Profit:",round(trade[1] - trade[3])*trade_amount)
		print("Trade Time:",datetime.strptime(trade[0][11:19],time_format) - datetime.strptime(trade[2][11:19],time_format))
		pnl.append(round(trade[1] - trade[3])*trade_amount)

	total_profit = sum(pnl)
	print("Total Profit:",total_profit)
	print("\n")
	#--Uncomment lines below to generate plots
	#plt.plot(pnl)
	##plt.ylabel('Thousands')
	#plt.title('Long Short Strategy - Profit:' + str(round(total_profit)))
	#plt.xlabel('Trades')
	#plt.show()


common_csv = '/1.csv'
months = ['03','04','05','06','07','08','09']

for day in months:
	path = '/home/marcus/pyalgo/lob_jsons' + folder + day + common_csv 
	trade_algo(path,threshold)




#--only runs if activate_multithres option is True
def main():
	common_csv = '/1.csv'
	months = ['03','04','05','06','07','08','09']
	thresholds = list(pl.frange(0.006,0.009,0.001))
	for thres in thresholds:
		for day in months:
			path = '/home/marcus/pyalgo/lob_jsons' + folder + day + common_csv 
			print("testing with threshold:",thres)
			trade_algo(inpath,thres)

if activate_multithres == True:
	main()








