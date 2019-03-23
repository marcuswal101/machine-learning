"""
   Python 3.6

   creates plots of implied volatilities and market volatilities using pandas dataframe

   Separate graphs are created using this program. Please follow the example below to create graphs for other data

   Usage:
           Keep outvol.csv file generated from the new_sabr_model.py in the same directory as this program

           Then run as:

                        python sabr_plot.py

"""

import matplotlib.pyplot as plt
import pandas as pd

dataframe_file = 'outvol.csv'
input_file = 'sample_format.xlsx'

xl = pd.ExcelFile(input_file)
df = xl.parse("tnotes")

#-----Get market volatilities to plot against
market_vol_0 = df.iloc[0][3:-1]
market_vol_1 = df.iloc[1][3:-1]
market_vol_2 = df.iloc[2][3:-1]
market_vol_3 = df.iloc[3][3:-1]
market_vol_4 = df.iloc[4][3:-1]

#------Get implied volatilities from sabr calibration to plot
df = pd.read_csv(dataframe_file,engine="python",index_col=False, header=1, delimiter='\;')

df_new_0 = df.iloc[0][2:-1]
df_new_1 = df.iloc[1][2:-1]
df_new_2 = df.iloc[2][2:-1]
df_new_3 = df.iloc[3][2:-1]
df_new_4 = df.iloc[4][2:-1]


#-------Example plot of implied volatilities versus market volatilities. Follow this method for the rest of data
plt.title('Sabr volatilities')
plt.plot(df_new_0.as_matrix(),label = 'implied volatility')
plt.plot(market_vol_0.as_matrix(),label = 'market volatility')
plt.legend()
plt.show()










