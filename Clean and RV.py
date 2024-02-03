import pandas as pd
import time
from datetime import timedelta
import sys
import os
import zipfile
import csv
import numpy as np
import matplotlib.pyplot as plt


def main():

    # df = readData()
    
    # All data already cleaned
    df = pd.read_csv('C:/Users/rafam/Downloads/EOR/Master/P3/Code/IBMdata.csv')
    
    # All RV's already in a dataframe
    RV = pd.read_csv('C:/Users/rafam/Downloads/EOR/Master/P3/Code/AllRV.csv')

    # Calculate RVsparse
    sparse = RVsparse(df)
    
    # Calculate RVdense and whats
    dense, wHats = RVdense(df)
    
    # Setting the parameters for the calculation of QV using Parzen Kernel
    H = 30
    variableH = 0
    if(variableH == 1):
        zeta = wHats/sparse
        zeta = pd.DataFrame(zeta, columns =['zeta'])
        zeta.fillna(method='ffill', inplace=True)
        zeta.fillna(method='bfill', inplace=True)
        zeta = zeta['zeta'].to_numpy()
    else:
        zeta = 0
    
    # Calling the function to calculate the QV using the Parzen Kernel
    QV, newH = efficient_kernel(df,zeta, variableH, H)
    df['DATE'] = pd.to_datetime(df['DATE'])
    returns = pd.DataFrame({'DATE': df['DATE'].unique()})
    returns['QV Parzen'] = QV
    returns['Hs'] = newH
    print(returns)

    # Printing and plotting the results
    plotRV(RV)
    
    # Calculating daily returns, using various open and closing measures
    log_returns_close,log_returns_co, log_returns_oc ,factor = log_returns(df)
    kernel_return = log_returns_close*100
    df['DATE'] = pd.to_datetime(df['DATE'])
    r = pd.DataFrame({'DATE': df['DATE'].unique()})
    r = r.drop(r.index[1])
    r.reset_index(drop=True, inplace=True)
   
    r['log returns close to close'] = log_returns_close
    r['kernel log returns'] = kernel_return
    r['close to open'] =log_returns_co
    r['open to close'] = log_returns_oc
    r.to_csv('returns.csv', index=False)    
    
def plotRV(df):
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    columns_to_plot = ['H=30', 'RVsparse', 'RVdense']
    for column in columns_to_plot:
        df[column] = pd.to_numeric(df[column])/10
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column], label=column, color='blue')
        plt.xlabel('Year')
        plt.ylabel('Realized Variance')
        plt.title(f'Realized Variance - {column}')
        plt.xticks(pd.date_range(start='2016-01-04', end='2024-01-01', freq='AS'))  # Annual frequency starting from 2016
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        plt.legend()
        plt.show()
        print(f'Mean of {column}: {np.mean(df[column])}')


def log_returns(df):
    # Convert 'time' to datetime type for sorting
    df['TIME_M'] = pd.to_datetime(df['TIME_M'], format='%H:%M:%S.%f')
    
    # Sort the DataFrame by 'day' and 'time'
    df = df.sort_values(by=['DATE', 'TIME_M'], ascending=[True, True])
    
    # Group by 'day' and select the last row (latest price for each day)
    latest_prices = df.groupby('DATE').tail(1)
    first_prices = df.groupby('DATE').head(1)
    # Reset the index 
    first_prices.reset_index(drop=True, inplace=True)
    latest_prices.reset_index(drop=True, inplace=True)
    
    log_price_close = np.log(latest_prices['MEDIAN_PRICE'])
    log_price_open = np.log(first_prices['MEDIAN_PRICE'])
    
    co_price_open = log_price_open[1:]
    co_price_closed = log_price_close[:-1]
    
    log_returns_co = ( co_price_closed -  co_price_open)*100 #close to open
    log_returns_oc = (log_price_open - log_price_close)*100 # open to close
    log_returns_close = np.diff(log_price_close)*100 #close to close
    
    factor = (np.var(log_returns_oc) + np.var(log_returns_co))/(np.var(log_returns_oc))
    
    print('f', factor)

    return log_returns_close,log_returns_co, log_returns_oc ,factor

def RVsparse(df):
    
    rolling_window_size = 1200 # 20 minutes rolling window
    all_seconds = pd.date_range('09:30:00', '16:00:00', freq='1S')
    
    sparse = []
    for _, dfDate in df.groupby('DATE'):
        dfDate['TIME_M'] = pd.to_datetime(dfDate['TIME_M']).round('1s')
        dfDate = dfDate.drop_duplicates(subset=['TIME_M','DATE'], keep='first')
        newTime = pd.DataFrame({'TIME_M': all_seconds})
        dfSparse = pd.merge(newTime, dfDate, on='TIME_M', how='left')
        
        dfSparse['MEDIAN_PRICE'].fillna(method='ffill', inplace=True)
        dfSparse['MEDIAN_PRICE'].fillna(method='bfill', inplace=True)
        
        realized_variances = []

        for i in range(rolling_window_size):
            # Step 1: Select every 20 minutes a row
            selected_rows_df = dfSparse.iloc[i::rolling_window_size, :]
            
            log_price = np.log(selected_rows_df['MEDIAN_PRICE'].values)
            log_returns = np.diff(log_price)*100
            squared_returns = np.square(log_returns)
            realized_variance = np.sum(squared_returns)
            realized_variances.append(realized_variance)

        realVariance = np.mean(realized_variances) 
        sparse.append(realVariance)

    return sparse    
  
def RVdense(df):
    
    q = 25
    dense = []
    wHats = []
    
    for _, dfDate in df.groupby('DATE'):
        
        realized_variances = []
        wiHat = []
        
        for i in range(q):
            # Step 1: Select every qth row
            selected_rows_df = dfDate.iloc[i::q, :]

            log_price = np.log(selected_rows_df['MEDIAN_PRICE'].values)
            log_returns = np.diff(log_price)*100
            squared_returns = np.square(log_returns)
            realized_variance = np.sum(squared_returns)
            realized_variances.append(realized_variance)
            
            nonzeroReturns = np.count_nonzero(log_returns)
            wiHat.append(realized_variance/(2*nonzeroReturns))

        realVariance = np.mean(realized_variances) 
        dense.append(realVariance)
        
        wHat = np.mean(wiHat)
        wHats.append(wHat)
        
    return dense, wHats
        
def parzen_kernel(x):
    condition1 = (0 <= np.abs(x)) & (np.abs(x) <= 0.5)
    condition2 = (0.5 < np.abs(x)) & (np.abs(x) <= 1)
    return np.where(condition1, 1 - 6 * x**2 + 6 * np.abs(x)**3, np.where(condition2, 2 * (1 - np.abs(x))**3, 0))

def gamma_h_vectorized(n, h, x):
    indices = np.arange(abs(h) + 1, n)
    
    result = np.sum(x[indices] * x[indices - abs(h)])
    return result

def efficient_kernel(df,zeta, variableH, H):
    result_per_day = []
    i = 0
    Hs = []

    for _, dfDate in df.groupby('DATE'):
        if(variableH == 1):
            H = int(3.5134*(zeta[i]**(4/5))*(len(dfDate)**(3/5)))
        Hs.append(H)

        i += 1
        n = len(dfDate) - 3
        result = 0

        log_price = np.log(dfDate['MEDIAN_PRICE'].values)
        log_price[0] = np.mean(log_price[:2])
        log_price[-1] = np.mean(log_price[-2:])
        logP = np.concatenate((log_price[:1], log_price[2:-2], log_price[-1:]))
        log_returns = np.diff(logP)
        x = log_returns * 100

        for h in range(-H, H + 1):
            result += parzen_kernel(h / (H + 1)) * gamma_h_vectorized(n, h, x)
            
        result_per_day.append(result)
            
    return result_per_day, Hs

def readData():
    df = readAll()
    df = clean(df)
    # print(df)
    df.to_csv('AllData.csv', index=False)
    return df

def clean(df):
    
    #P1  Delete entries with a time stamp outside the 9:30 am–4 pm window when the exchange is open
    start_time = pd.to_datetime('09:30:00')
    end_time = pd.to_datetime('16:00:00')
    df['Time'] = pd.to_datetime(df['TIME_M'])   
    df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    df.reset_index(drop=True, inplace=True)
        
    #P2 Delete entries with a bid, ask or transaction price equal to zero.
    df = df[df['PRICE'] != 0]
    
    #P3 Retain entries originating from the most frequent exchange. Delete other entries
    most_freq = df['EX'].value_counts()[:1].index.tolist()
    # print(most_freq[0])
    df.drop(df[df['EX'] != most_freq[0]].index, inplace=True)
    
    #P3 Check
    checkExchangeFrequency = 0
    
    if checkExchangeFrequency == 1:
        df['Datum'] = pd.to_datetime(df['DATE'])

        # Extract month from 'DATE' and create a new column 'MONTH'
        df['MONTH'] = df['Datum'].dt.to_period('M')
        df.drop('Datum', axis=1, inplace=True)

        # Group by 'MONTH' and 'EXCHANGE', and count the trades
        exchange_counts = df.groupby(['MONTH', 'EX']).size().reset_index(name='TRADE_COUNT')

        # Sort by 'TRADE_COUNT' in descending order for each month
        exchange_counts.sort_values(['MONTH', 'TRADE_COUNT'], ascending=[True, False], inplace=True)

        # Select the top 3 exchanges for each month
        top_3_exchanges_per_month = exchange_counts.groupby('MONTH').head(3)

        # Display the count of trades for the three most frequent exchanges for each month
        print(top_3_exchanges_per_month)
        

    #T1 Delete entries with corrected trades. (Trades with a Correction Indicator, CORR != 0).
    df = df[df['TR_CORR'] == 0]
    
    #T2 (and P1 since T is excluded) Delete entries with abnormal Sale Condition. (Trades where COND includes a T or Z)
    df = df[~((df['TR_SCOND'].str.contains('T|Z')) | (pd.isna(df['TR_SCOND'])))]
     
    #T3  If multiple transactions have the same time stamp, use the median price
    df['TIMESTAMP'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME_M'])
    df = df.sort_values(by='TIMESTAMP')
    
    # Make a new column median price for trades with the same timestamp
    df['MEDIAN_PRICE'] = df.groupby('TIMESTAMP')['PRICE'].transform('median')
    df.drop('TIMESTAMP', axis=1, inplace=True)

    # Only keep the first trade on a duplicate time stamp, since median price is the same
    df = df.drop_duplicates(subset=['TIME_M','DATE'], keep='first')
    
    #T4  Delete entries for which the mid-quote deviated by more than 10 mean absolute deviations
    # from a rolling centred median (excluding the observation under consideration) of 50
    # observations (25 observations before and 25 after)
    df = rollingMedian(df)

    # Reset the index of the filtered dataframe
    df.reset_index(drop=True, inplace=True)
    
    return df

def rollingMedian(df):
    window_size = 50
    
    # Calculate the rolling centered median and MAD
    rolling_median = df['MEDIAN_PRICE'].rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = df['MEDIAN_PRICE'].rolling(window=window_size, center=True, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.median(x))))
    
    # Define the threshold for filtering
    threshold = 10 * rolling_mad
    
    # Filter the dataframe based on the deviation threshold
    filtered_df = df.loc[np.abs(df['MEDIAN_PRICE'] - rolling_median) <= threshold]
    
    return filtered_df

def readAll():
    
    #Read all files into one bid dataframe, takes 5-10 mintues to read
    zip_files_directory = 'C:/Users/rafam/Downloads/EOR/Master/P3/Data/'
    combined_df = pd.DataFrame()
    
    for zip_file_name in os.listdir(zip_files_directory):
        if zip_file_name.endswith('.zip'):
            zip_file_path = os.path.join(zip_files_directory, zip_file_name)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                csv_file_name = zip_ref.namelist()[0]
                
                with zip_ref.open(csv_file_name) as csv_file:
                    df = pd.read_csv(csv_file)
                    combined_df = combined_df.append(df, ignore_index=True)
    
    return combined_df
  

###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:",timedelta(seconds=end_time - start_time))
