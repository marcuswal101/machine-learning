###lob data from json to csv for the 38 pairs limit order book. csv export contains Bidprice, Bidvolume, Askprice, and Askvolume as headers###

import csv, json
import pandas as pd
import numpy as np

concatenate = False

#filepath = '/home/marcus/Desktop/zorro/btcusdtrade.json'  
#outpath = '/home/marcus/Desktop/zorro/btcusdtrade.csv'

#filepath = '/home/marcus/pyalgo/btcusd_lob.json'  
#outpath = '/home/marcus/pyalgo/btcusd_lob_date.csv'

#filepath = '/home/marcus/pyalgo/ETHUSD_lob.json'  
#outpath = '/home/marcus/pyalgo/ETHUSD_lob_date.csv'

# folder = '/ADAETH/'
#folder = '/EOSETH/'
#folder = '/NEOETH/'

#folder = '/BTCUSDT/'
# folder = '/ETHUSDT/'
folder = '/LTCETH/'

filepath = '/home/marcus/pyalgo/lob_jsons' + folder  
outpath = '/home/marcus/pyalgo/lob_jsons' + folder


def jsontocsv(filepath,outpath):
	data = pd.read_json(filepath)

	data['E'] = pd.to_datetime(data['E'], unit='ms') #convert unix time

	new_b = data[['b']] #bid data
	new_a = data[['a']] #ask data

	entries = new_b.shape[0] #data count

	bidprice = []
	bidvolume = []
	askprice = []
	askvolume = []
	for row in range(entries):
		try:
			bidprice.append(list(new_b.iloc[row][0].keys())[0].replace('_','.'))
			bidvolume.append(list(new_b.iloc[row][0].values())[0])
		except Exception as e:
			bidprice.append(np.NaN)
			bidvolume.append(np.NaN)		
	

	for row in range(entries):
		try:
			askprice.append(list(new_a.iloc[row][0].keys())[0].replace('_','.'))
			askvolume.append(list(new_a.iloc[row][0].values())[0])
		except Exception as e:
			askprice.append(np.NaN)
			askvolume.append(np.NaN)


	#print(bidprice)
	new_b['Bidprice'] = bidprice
	new_b['Bidvolume'] = bidvolume
	new_a['Askprice'] = askprice
	new_a['Askvolume'] = askvolume

	final_df = new_b[['Bidprice','Bidvolume']]
	final_df['Askprice'] = new_a[['Askprice']]
	final_df['Askvolume'] = new_a[['Askvolume']]
	final_df['Time'] = data[['E']]

	#--drop NaN values and zero values
	final_df = final_df.dropna()
	final_df.to_csv(outpath, index = False)


common_json = '/1.json'
common_csv = '/1.csv'
months = ['03','04','05','06','07','08','09']
all_csv_paths = [] #to use later with pandas concatenation
for month in months:
	new_filepath = filepath + month + common_json
	new_outpath = outpath + month + common_csv
	all_csv_paths.append(new_outpath)
	jsontocsv(new_filepath,new_outpath)


#concatenating csv files into a single file - Default set to False
if concatenate == True:
	concatenate_path = filepath + 'concatenated.csv'
	combined_csv = pd.concat( [ pd.read_csv(f) for f in all_csv_paths ] )
	combined_csv.to_csv(concatenate_path, index=False )





