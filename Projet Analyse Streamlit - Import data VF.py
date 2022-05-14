# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 20:27:36 2021

@author: Julien
"""
import pandas as pd
import concurrent.futures
import datetime
import yfinance as yf
import os
import pickle
import time
global ticker_exception

Fondamental_data_path='D:\\Julien\\Investissement\\Streamlit\\Histo Data Fondamental'
Stock_prices_data_path='D:\\Julien\\Investissement\\Streamlit\\Histo Data Stock prices'

Final_data_file='D:\Julien\Investissement\Symboles\Liste_Ticker_Final_with_IPO.xlsx'
Final_data_file_info='D:\Julien\Investissement\Symboles\Stock World Tickers.xlsx'
Index_list=['Cryptocurrencies','RUSSEL 1000','CAC Index','S&P500','NASDAQ 100','AEX Index','AS25 Index','BEL20 Index','CH30 Index','DAX Index','FTSEMIB Index','HDAX Index','HEX25 Index','IBEX Index','NEY Index','OMX Index','OMXC25 Index','SBF120 Index','SMI Index','SPTSX60 Index','NASDAQ 1','NASDAQ 2','NASDAQ 3','NASDAQ 4','NYSE 1','NYSE 2']
Index_list_fonda=['RUSSEL 1000','CAC Index','S&P500','NASDAQ 100','AEX Index','AS25 Index','BEL20 Index','CH30 Index','DAX Index','FTSEMIB Index','HDAX Index','HEX25 Index','IBEX Index','NEY Index','OMX Index','OMXC25 Index','SBF120 Index','SMI Index','SPTSX60 Index','NASDAQ 1','NASDAQ 2','NASDAQ 3','NASDAQ 4','NYSE 1','NYSE 2']# Index_list=['Cryptocurrencies']
Crypto_list=['Cryptocurrencies']
# Index_list=['CH30 Index','DAX Index','FTSEMIB Index','HDAX Index','HEX25 Index','IBEX Index','NEY Index','OMX Index','OMXC25 Index','SBF120 Index','SMI Index','SPTSX60 Index','NASDAQ']
# Index_list_fonda=['NYSE 1','NYSE 2']
####################################################################################################################
#Other functions used

# Df_final=pd.read_excel(Final_data_file, index_col=0)
# Df_info=pd.read_excel(Final_data_file_info, index_col=0)
# print(Df_info[Df_info['Stock Exchange']=='BEL20 Index'].index.to_list())

def Import_data_ticker(file,Index):
    import pandas as pd
    Df_final=pd.read_excel(file,'Sheet1', index_col=0)
    Df_info=pd.read_excel(Final_data_file_info, index_col=0)
    Df_final=Df_final[Df_final['Ticker'].isin(Df_info[Df_info['Stock Exchange']==Index].index.to_list())]
    Final_Excel_data_dic=Df_final["Ticker"].to_list()
    
    return Final_Excel_data_dic

def max_date_imported():
    files=os.listdir(Stock_prices_data_path)
    max_date=max(files, key=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'))
    return max_date
  
def Today_date():
    return datetime.datetime.today().strftime('%Y-%m-%d')

def Dossier_fondamental_data_creation(date):
    try:
       os.mkdir(Fondamental_data_path+'\\'+Today_date())
    except:
        pass

def Dossier_stock_prices_data_creation(date):
    try:
       os.mkdir(Stock_prices_data_path+'\\'+Today_date())
    except:
        pass

def Import_Fondamental_data_test(ticker_list):
    return yf.Ticker(ticker_list).info

####################################################################################################################
                                            #Import data from yahoo finance

###################################################################
#Import historical stock prices function incrémental --> Fail car très lent

# def Import_stock_prices_incrementation(Symbols,date,Int = "1d"):
#     # inputs
    
#     infile_m=open(Stock_prices_data_path+'\\'+date_last_histo+'\\'+date_last_histo+' '+index+' - Market data','rb')
#     infile_p=open(Stock_prices_data_path+'\\'+date_last_histo+'\\'+date_last_histo+' '+index+' - Prices','rb')
#     infile_y=open(Stock_prices_data_path+'\\'+date_last_histo+'\\'+date_last_histo+' '+index+' - Yields','rb')
    
#     DataFrame_m=pickle.load(infile_m, encoding='latin1')
#     DataFrame_p=pickle.load(infile_p, encoding='latin1')
#     DataFrame_y=pickle.load(infile_y, encoding='latin1')
    
#     DataFrame_m_retraité=DataFrame_m.drop(DataFrame_m.tail(3).index)
#     DataFrame_p_retraité=DataFrame_p.drop(DataFrame_p.tail(3).index)
#     DataFrame_y_retraité=DataFrame_y.drop(DataFrame_y.tail(3).index)
    
#     max_date_histo=max(DataFrame_m_retraité.index)
#     start_date=max_date_histo.strftime('%Y-%m-%d')
    
#     DataFrame_m_retraité=DataFrame_m_retraité.drop(DataFrame_m_retraité.tail(1).index)
#     DataFrame_p_retraité=DataFrame_p_retraité.drop(DataFrame_p_retraité.tail(1).index)
#     DataFrame_y_retraité=DataFrame_y_retraité.drop(DataFrame_y_retraité.tail(1).index)
    
#     Stock_Tickers=""
#     for stock in Symbols:
#         Stock_Tickers += " " 
#         Stock_Tickers += stock

#     stock_data= yf.download(tickers=Stock_Tickers,group_by = 'ticker',start=start_date,interval = Int,auto_adjust = False,threads = 8,proxy = None)
    
#     frames=[DataFrame_m_retraité,stock_data]
#     df_m=pd.concat(frames)
#     outfile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Market data','wb')
#     pickle.dump(df_m, outfile)
#     outfile.close()
    
#     DataFrame_p=stock_data.iloc[:, stock_data.columns.get_level_values(1)=='Close']
#     DataFrame_p.columns = DataFrame_p.columns.droplevel(1)
#     frames=[DataFrame_p_retraité,DataFrame_p]
#     df_p=pd.concat(frames)
#     outfile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Prices','wb')
#     pickle.dump(df_p, outfile)
#     outfile.close()

#     DataFrame_y=DataFrame_y.pct_change().iloc[1:]
#     frames=[DataFrame_y_retraité,DataFrame_y]
#     df_y=pd.concat(frames)
#     outfile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Yields','wb')
#     pickle.dump(df_y, outfile)
#     outfile.close()

###################################################################
#Import historical stock prices function

def Import_stock_prices(Symbols,date,period_history="10y",Int = "1d"):
     # inputs
     Stock_Tickers=""
     for stock in Symbols:
         Stock_Tickers += " " 
         Stock_Tickers += stock
     # print(Stock_Tickers)
     stock_data= yf.download(tickers=Stock_Tickers,period=period_history,group_by = 'ticker',interval = Int,auto_adjust = False,threads = 8,proxy = None)
     outfile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Market data','wb')
     pickle.dump(stock_data, outfile)
     outfile.close()
     
     DataFrame=stock_data.iloc[:, stock_data.columns.get_level_values(1)=='Close']
     DataFrame.columns = DataFrame.columns.droplevel(1)
     outfile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Prices','wb')
     pickle.dump(DataFrame, outfile)
     outfile.close()
     
     return stock_data
 
def Rendement_Fermeture_Export_historical_data(date):
    
    infile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Prices','rb')
    DataFrame=pickle.load(infile, encoding='latin1')
    DataFrame=DataFrame.pct_change().iloc[1:]
    outfile=open(Stock_prices_data_path+'\\'+Today_date()+'\\'+date+' '+index+' - Yields','wb')
    pickle.dump(DataFrame, outfile)
    outfile.close()
    
    return DataFrame

###################################################################
#Import fondamental data function

def Import_data_fondamental(filename,Symbols,date):
    i=0
    tickers_data= {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor: 
        ticker_object = executor.map(Import_Fondamental_data_test,Symbols)

        
    for infos in ticker_object:
        
        symbol=Symbols[i]
        i+=1
        try:
            temp = pd.DataFrame.from_dict(infos, orient="index")
            temp.reset_index(inplace=True)
            temp.columns = ["Attribute", "Recent"]
            #modif 24/07/2021
            # temp=temp[temp['Attribute'].isin(['exchange','country','currency','longName','sector','industry','priceToBook','fiveYearAvgDividendYield','dividendRate','trailingAnnualDividendYield','enterpriseToEbitda','enterpriseToRevenue','enterpriseValue','marketCap','volume','regularMarketVolume','averageDailyVolume10Day'])]
        except:
            temp = pd.DataFrame(columns=["Attribute", "Recent"])
            print("###########"+symbol+"##############")

        tickers_data[symbol] = temp
        
    combined_data = pd.concat(tickers_data)
    combined_data = combined_data.reset_index()
    combined_data.to_excel(Fondamental_data_path+'\\'+Today_date()+'\\'+date+' '+index+".xlsx")  
    
    del combined_data["level_1"] # clean up unnecessary column
    combined_data.columns = ["Ticker", "Attribute", "Recent"] # update column names
    
    outfile=open(Fondamental_data_path+'\\'+date+'\\'+filename,'wb')
    pickle.dump(combined_data, outfile)
    outfile.close()

    all_data=combined_data.pivot(index='Ticker', columns='Attribute', values='Recent')
    # df1=all_data.sort_values(by=['sector','industry','priceToBook'])
    all_data.to_excel(Fondamental_data_path+'\\'+Today_date()+'\\'+date+' '+index+".xlsx")  
    
####################################################################################################################
#Execution of functions

###################################################################
#input in common
input_date=Today_date()
start_time1 = time.time()


# for index in Crypto_list:
    
#     print(index+" - Stock data - Import Process")
    
#     dic_ticker=Import_data_ticker(Final_data_file,index)

#     ###################################################################
#     #Execution of Histo Stock Prices data import
    
#     start_time = time.time()
    
#     Dossier_stock_prices_data_creation(input_date)
    
#     df_prices=Import_stock_prices(dic_ticker,input_date)
#     df_yields=Rendement_Fermeture_Export_historical_data(input_date)


#     ###################################################################

# max_date_imported()

# index='CAC Index'

# for index in Index_list:
#     print(index+" - Stock data - Import Process")
    
#     dic_ticker=Import_data_ticker(Final_data_file,index)
    
#     start_time = time.time()
    
#     date_last_histo=max_date_imported()
#     Dossier_stock_prices_data_creation(input_date)
    
#     Import_stock_prices_incrementation(dic_ticker,input_date)
#     time.sleep(2)


for index in Index_list:
    
    print(index+" - Stock data - Import Process")
    
    dic_ticker=Import_data_ticker(Final_data_file,index)

    ###################################################################
    #Execution of Histo Stock Prices data import
    
    start_time = time.time()
    
    Dossier_stock_prices_data_creation(input_date)
    
    df_prices=Import_stock_prices(dic_ticker,input_date)
    df_yields=Rendement_Fermeture_Export_historical_data(input_date)
    
    time.sleep(10)
    
    ###################################################################
    #Execution of fondamental data import

for index in Index_list_fonda:
    
    start_time = time.time() 
    
    dic_ticker=Import_data_ticker(Final_data_file,index)
    
    print(index+" - Fondamental data - Import Process")
    
    Dossier_fondamental_data_creation(input_date)
    
    start_time = time.time()
    
    time.sleep(60)
    
    Import_data_fondamental(index+"-Fondamental data",dic_ticker,input_date)
    
    print("Temps d'execution : %s secondes" % (time.time() - start_time))
    
print("Temps d'execution : %s secondes" % (time.time() - start_time1))
    