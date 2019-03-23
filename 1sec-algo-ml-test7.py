"""implementing supervised machine learning algorithm on the oxford thesis strategy"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import pylab as pl
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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
plot = False #If True then plots the trade profits
threshold = 0.006 #btc/usd and eth/usd
strategy = 'A' # or 'B'. However, A is recommeded for cryptocurrencies 
trade_amount = 1 # 1 BTC or ETH etc
trade_fees = 0 
test_percent = 0.4


# inpath = '/home/marcus/pyalgo/btcusd_lob_date.csv' # 1 day data
# inpath = '/home/marcus/pyalgo/ETHUSD_lob_date.csv' # 1 day data

# folder = '/ADAETH/' # 7 days data
# folder = '/EOSETH/' # 7 days data
# folder = '/NEOETH/' # 7 days data
# folder = '/BTCUSDT/' # 7 days data
# folder = '/BTCUSDT27/'
# folder = '/ETHUSDT/' # 7 days data
# folder = '/LTCETH/'
# inpath = '/home/marcus/pyalgo/lob_jsons' + folder + 'concatenated.csv' 

assets = ['/BTCUSDT/','/ETHUSDT/','/ADAETH/','/EOSETH/','/NEOETH/']


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
		# --regression strategy A: midresponse (y) and VOI(s) (x) and width and aggressor features
		result = sm.ols(formula="midresponse ~ VOI + VOI_1 + VOI_2 + VOI_3 + VOI_4 + VOI_5 + width", data=df).fit()
		summary = result.summary()
		midr_fit = result.fittedvalues #dataframe of the fitted value of midresponse
		midr_avg = result.fittedvalues.mean() #average of the predicted mid response value. Use it to set threshold
		# print('Average MidResponse is:',midr_avg)
	else:
		pass
	#--running trade simulation
	df['midr_fit'] = midr_fit #add the fitted y values to the dataframe
	df = df.reset_index() #reset index values to start from zero

	#--implementing machine learning withf eatures to predict price: VOI ,VOI_1 ,VOI_2 ,VOI_3 ,VOI_4 ,VOI_5 ,width, and spread
	# X_data = df[['VOI','VOI_1','VOI_2','VOI_3','VOI_4','VOI_5','width']]
	X_data = df[['VOI','VOI_1','VOI_2','VOI_3','VOI_4','VOI_5','width','spread']]
	# Y_data = df[['midr_fit']]
	Y_data = df[['midprice']]
	#--making test data	
	X_train = X_data[:-1*int(df.shape[0]*test_percent)]
	# print(X_train)
	X_test = X_data[-1*int(df.shape[0]*test_percent):]
	# print(X_test)
	#--making target data
	Y_train = Y_data[:-1*int(df.shape[0]*test_percent)]
	Y_test = Y_data[-1*int(df.shape[0]*test_percent):]
	# Create linear regression object
	regr = linear_model.LinearRegression()
	# Train the model using the training sets
	regr.fit(X_train, Y_train)
	# Make predictions using the testing set
	Y_pred = regr.predict(X_test)
	
	#--check if test and prediction values have same length
	if len(Y_pred) == len(Y_test):
		print(True)

	#--implement trade strategy using the machine learning predictions

	buyprices = []
	sellprices = []
	buytime = []
	selltime = []
	backtest_data = df[-1*int(df.shape[0]*test_percent):]
	backtest_data = backtest_data.reset_index()
	for index, row in backtest_data.iterrows(): #loop over the backtest data

		# print(index)
		# break
		if row["Askprice"] < Y_pred[index]:
			# print("ASK")
			buyprice = row["Askprice"]
			buy_time = row["Time"]
			buyprices.append(buyprice)
			buytime.append(buy_time)
		elif row["Bidprice"] > Y_pred[index]:
			# print("BID")
			sellprice = row["Bidprice"]
			sell_time = row["Time"]
			sellprices.append(sellprice)
			selltime.append(sell_time)
		else:
			pass

	pnl = []
	time_format = "%H:%M:%S"
	for trade in zip(selltime,sellprices,buytime,buyprices):
		if trade[2] < trade[0]:
			# print("Time:%s Short at Price:%s Time:%s Long at Price:%s"%(trade[0],trade[1],trade[2],trade[3]))
			# print("Profit:",round(trade[1] - trade[3])*trade_amount)
			# print("Trade Time:",datetime.strptime(trade[0][11:19],time_format) - datetime.strptime(trade[2][11:19],time_format))
			pnl.append((trade[1] - trade[3])*trade_amount)

	total_profit = round(sum(pnl))
	print("Total Profit:",total_profit)
	print("\n")
	#--Uncomment lines below to generate plots
	if plot == True:
		plt.plot(pnl)
		#plt.ylabel('Thousands')
		plt.title('Long Short Strategy - Profit:' + str(round(total_profit)))
		plt.xlabel('Trades')
		plt.show()


common_csv = '/1.csv'
concatenated_csv = '/concatenated.csv'
# months = ['03','04','05','06','07','08','09']
# path = '/home/marcus/pyalgo/lob_jsons' + folder + concatenated_csv 
# trade_algo(path,threshold)

for folder in assets:
	print(folder)
	path = '/home/marcus/pyalgo/lob_jsons' + folder + concatenated_csv 
	trade_algo(path,threshold)
# profit = trade_algo(path,threshold)
# print(profit)











