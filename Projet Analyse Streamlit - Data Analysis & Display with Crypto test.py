# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 19:56:33 2021

@author: Julien
"""

import streamlit as st
import pandas as pd
from datetime import datetime,timedelta
import pickle
import time
import talib
from TA_technical_patterns import candlestick_patterns
import plotly.graph_objects as go
# import pyfolio as pf
import seaborn as sns
import pandas_ta as ta
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
# import altair as alt
# from functions_streamlit import *

Fondamental_data_path='D:\\Julien\\Investissement\\Streamlit\\Histo Data Fondamental'
Stock_prices_data_path='D:\\Julien\\Investissement\\Streamlit\\Histo Data Stock prices'
Crypto_data_file='D:\\Julien\\Investissement\\Symboles\\Crypto Tickers EUR.xlsx'
# index='S&P500'
Index_list=['Cryptocurrencies','CAC Index','S&P500','NASDAQ 100','RUSSEL 1000','AEX Index','AS25 Index','BEL20 Index','CH30 Index','DAX Index','FTSEMIB Index','HDAX Index','HEX25 Index','IBEX Index','NEY Index','OMX Index','OMXC25 Index','SBF120 Index','SMI Index','SPTSX60 Index','NASDAQ 1','NASDAQ 2','NASDAQ 3','NASDAQ 4']
# Index_list=['Cryptocurrencies']
# Index_list=['CAC Index','S&P500','NASDAQ 100','RUSSEL 1000']
# Index_list=['Cryptocurrencies']

####################################################################################################################
#Function already existing in Import data

def Today_date(days_to_subtract):
    date_import=datetime.today() - timedelta(days=days_to_subtract)
    date_import=date_import.strftime('%Y-%m-%d')
    return date_import

def Fig_candlestick_setup(fig,position=0):
    fig.data[position].decreasing.fillcolor = "darkred"
    fig.data[position].decreasing.line.color = "darkred"
    
    fig.data[position].increasing.fillcolor = "darkgreen"
    fig.data[position].increasing.line.color = "darkgreen"    
    
    fig.data[position].decreasing.line.width = 1
    fig.data[position].increasing.line.width = 1

def fillcol(label):
    if label >= 1:
        return 'rgba(26,150,65,0.3)'
    else:
        return 'rgba(255, 0, 0, 0.3)'
    
# date=Today_date()
date=Today_date(0)
date_fonda=Today_date(0)#'2021-11-27'

####################################################################################################################
#Execution of cash functions

start_time = time.time()

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_fondamental(indice):
    df_fondamental=pickle.load(open(Fondamental_data_path+'\\'+date_fonda+'\\'+indice+'-Fondamental data','rb'), encoding='latin1')
    return df_fondamental

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_market_data(indice):
    df_market_data=pickle.load(open(Stock_prices_data_path+'\\'+date+'\\'+date+' '+indice+' - Market data','rb'), encoding='latin1')    
    return df_market_data

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_prices_data(indice):
    df_prices_data=pickle.load(open(Stock_prices_data_path+'\\'+date+'\\'+date+' '+indice+' - Prices','rb'), encoding='latin1')    
    return df_prices_data

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_yields_data(indice):
    df_yields_data=pickle.load(open(Stock_prices_data_path+'\\'+date+'\\'+date+' '+indice+' - Yields','rb'), encoding='latin1')    
    return df_yields_data


####################################################################################################################
#Display Streamlit

start_time = time.time()

st.set_page_config(layout='wide')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

####################################################################################################################
#Display Streamlit function


                # except:
                #     pass

#######################################################
#Load Dataframes

df_fondamental_temp= {}
df_market_temp= {}
df_prices_temp= {}
df_yields_temp= {}

for index in Index_list:
    df_market_temp[index]=load_market_data(index)
    df_prices_temp[index]=load_prices_data(index)
    df_yields_temp[index]=load_yields_data(index)
    if index!='Cryptocurrencies':
        df_fondamental_temp[index]=load_fondamental(index)

print("--- %s seconds part load data---" % (time.time() - start_time))

@st.cache(suppress_st_warning=True,allow_output_mutation=True,show_spinner=False)
def concat_fonda():
    if df_fondamental_temp!={}:
        df_concat=pd.concat(df_fondamental_temp) 
    else: 
        df_concat=[]
    return df_concat

@st.cache(suppress_st_warning=True,allow_output_mutation=True,show_spinner=False)
def concat_list_df(liste):
    df_concat=pd.concat(liste,axis=1)
    return df_concat


df_fondamental= concat_fonda()
df_market= concat_list_df(df_market_temp)
df_prices= concat_list_df(df_prices_temp)
df_yields= concat_list_df(df_yields_temp)

# df_fondamental= pd.concat(df_fondamental_temp)
# df_market= pd.concat(df_market_temp,axis=1)
# df_prices= pd.concat(df_prices_temp,axis=1)
# df_yields= pd.concat(df_yields_temp,axis=1)

print("--- %s seconds part concatenate---" % (time.time() - start_time))

#Download mapping ticker cryptocurrencies
Df_crypto_name=pd.read_excel(Crypto_data_file, index_col=0)

print("--- %s seconds part load crypto---" % (time.time() - start_time))

# dict_crypto_ticker_to_name=dict(zip(Df_crypto_ticker_to_name['Ticker'],Df_crypto_ticker_to_name['Name']))

# df_test=df_market.loc[:, df_market.columns.get_level_values(1).isin(['ACA.PA'])]

####################################################################################################################

# df_prices_focus=df_prices.copy()

Index_sorted=sorted(Index_list)
Index_sorted.append("All")
list_tickers=list(df_market.columns.get_level_values(1).unique())
patterns_sorted=sorted(list(candlestick_patterns.values()))
Industry_name='industry'
Sector_name='sector'

# df_market=df_market[df_market.index>'23-09-2020']
# df_prices=df_prices[df_prices.index>'23-09-2020']
# df_yields=df_yields[df_yields.index>'23-09-2020']

# df_market=df_market[df_market.index>'01-01-2019']
# df_prices=df_prices[df_prices.index>'01-01-2019']
# df_yields=df_yields[df_yields.index>'01-01-2019']

max_date = max(df_market.index)#.strftime("%d-%m-%Y")
min_date = min(df_market.index)#.strftime("%d-%m-%Y")
default_date=datetime.today() - timedelta(days=365)

Beg_date = st.date_input("Pick a date", value=default_date,min_value=min_date, max_value=datetime.today())
Beg_date=Beg_date.strftime("%Y-%m-%d")

df_market=df_market[df_market.index>Beg_date]
df_prices=df_prices[df_prices.index>Beg_date]
df_yields=df_yields[df_yields.index>Beg_date]

Asset  = st.selectbox("Choose your Asset type", ['','Stocks','Crypto'])

print("--- %s seconds part sort---" % (time.time() - start_time))
page=''

if Asset:
    if Asset=='Stocks':
        df_fondamental = df_fondamental.astype(str)
        df_market=df_market[df_market.index>'01-01-2019']
        df_prices=df_prices[df_prices.index>'01-01-2019']
        df_yields=df_yields[df_yields.index>'01-01-2019']
        Index  = st.multiselect("Choose your Market Index", Index_sorted)
        if "All" in Index:
            Index =Index_list
        dict_name_to_ticker=df_fondamental[(df_fondamental['Attribute']=='shortName') & (df_fondamental.index.get_level_values(0).isin(Index))][['Ticker','Recent']].set_index('Recent').to_dict()['Ticker']
        dict_ticker_to_name=df_fondamental[(df_fondamental['Attribute']=='shortName') & (df_fondamental.index.get_level_values(0).isin(Index))][['Ticker','Recent']].set_index('Ticker').to_dict()['Recent']
        df_fondamental_Index=df_fondamental.loc[Index]
        Sector_list=df_fondamental_Index[df_fondamental_Index['Attribute']==Sector_name]['Recent'].drop_duplicates().tolist()
        df_fondamental_Index = df_fondamental_Index.astype(str)
        page = st.sidebar.selectbox("Choose your analysis view :", ["Relative Value", "Focus", "Technical Analysis"])
        st.header(page)
        filtered = {k: v for k, v in dict_ticker_to_name.items() if v is not None}
        dict_ticker_to_name.clear()
        dict_ticker_to_name.update(filtered)
        df_prices=df_prices.iloc[:,df_prices.columns.get_level_values(0).isin(Index)]
        # st.dataframe(df_prices.tail(200))
        df_market=df_market.loc[:, df_market.columns.get_level_values(0).isin(Index)]
    else:
        Index  = ['Cryptocurrencies']
        dict_ticker_to_name=dict(zip(Df_crypto_name.index,Df_crypto_name['Name']))
        dict_name_to_ticker=dict(zip(Df_crypto_name['Name'],Df_crypto_name.index))
        page = st.sidebar.selectbox("Choose your analysis view :", ["Focus", "Technical Analysis"])
        st.header(page)    
        filtered = {k: v for k, v in dict_ticker_to_name.items() if v is not None}
        dict_ticker_to_name.clear()
        dict_ticker_to_name.update(filtered)
        df_prices=df_prices.iloc[:,df_prices.columns.get_level_values(0).isin(Index)]
        df_market=df_market.loc[:, df_market.columns.get_level_values(0).isin(Index)]

print("--- %s seconds part if---" % (time.time() - start_time))
    
if page == "Relative Value" :
    
    Sector  = st.multiselect("Choose your Sector :", Sector_list)
    
    # st.write(df_fondamental_Index)
    Sector_Ticker=df_fondamental_Index[df_fondamental_Index['Attribute']==Sector_name][['Ticker','Recent']][df_fondamental_Index['Recent'].isin(Sector)]['Ticker'].tolist()
    # st.write(df_fondamental_Index[df_fondamental_Index.index.isin(df_fondamental_test)])
    if Sector:
        Industry_list=df_fondamental_Index[df_fondamental_Index.Ticker.isin(Sector_Ticker)][df_fondamental_Index['Attribute']==Industry_name]['Recent'].drop_duplicates().tolist()
        Industry_list_sorted=sorted(Industry_list)
        Industry_list_sorted.append("All")
        # st.write(df_fondamental_test)
        # st.write(Industry_name)
        # st.write(df_fondamental_Index[df_fondamental_Index.index.get_level_values(0).isin(df_fondamental_test)])#[df_fondamental_Index['Attribute']==Industry_name])
        # st.write(Industry_list)
        Industry  = st.multiselect("Choose your Industry :", Industry_list_sorted)
        if "All" in Industry:
            Industry =Industry_list
        
        # df_fondamental_Index[df_fondamental_Index['Attribute']==Sector_name][['Ticker','Recent']][df_fondamental_Index['Recent'].isin(Sector)]['Ticker'].tolist()
        if Industry:
            df_fondamental_Index_filter_ticker=df_fondamental_Index[(df_fondamental_Index.Ticker.isin(Sector_Ticker)) & (df_fondamental_Index['Attribute']==Industry_name)][['Ticker','Recent']][df_fondamental_Index['Recent'].isin(Industry)]['Ticker'].tolist()
            
            # st.write(df_fondamental_Index[(df_fondamental_Index.Ticker.isin(Sector_Ticker)) & (df_fondamental_Index['Attribute']==Industry_name)])
            
            # st.write(df_fondamental_Index_filter_ticker)
            
            df_fondamental_analyse=df_fondamental_Index[df_fondamental_Index.Ticker.isin(df_fondamental_Index_filter_ticker)]
            
            # st.write(df_fondamental_analyse)
            df_fondamental_analyse=df_fondamental_analyse.pivot_table(values='Recent', index='Ticker', columns='Attribute', aggfunc='first')
            
            #########################################################################
            #To identify field available by Yfinance
            # st.dataframe(df_fondamental_analyse,width=1500,height=2000)
            #########################################################################
            # st.write(list(df_fondamental_analyse.index.values))
            
            Fond_metrics_list=['beta','currency','financialCurrency','debtToEquity','dividendYield','earningsGrowth','forwardEps','forwardPE','grossMargins','operatingMargins','PayoutRatio','pegRatio','pricetoBook','PriceToSalesTrailing12Months','ProfitMargins','quickRatio','returnOnAsset','returnOnEquity','RevenueGrowth','RevenuePerShare','trailingAnnualDividendYield','trailingEps','trailingPE']
            
            Fond_metrics_list_profitability=['earningsGrowth','earningsQuarterlyGrowth','grossMargins','operatingMargins','profitMargins','revenuePerShare']
            Fond_metrics_list_price=['priceToBook','trailingPE','priceToSalesTrailing12Months','trailingEps','forwardEps','forwardPE','pegRatio']
            Fond_metrics_list_BS=['quickRatio','currentRatio','enterpriseToRevenue','totalCashPerShare','debtToEquity','returnOnAssets','returnOnEquity']
            Fond_metrics_list_Market=['beta','sharesPercentSharesOut','shortPercentOfFloat','shortRatio']
            Fond_metrics_list_Absolute=['sharesOutstanding','sharesShortPriorMonth','totalCash','totalAssets','totalDebt','totalRevenue']
            
            Fond_metrics_list=Fond_metrics_list_profitability+Fond_metrics_list_price+Fond_metrics_list_BS+Fond_metrics_list_Market+Fond_metrics_list_Absolute
            
            # for e in Fond_metrics_list:
            #     if Fond_metrics_list.count(e) == 0:
                    
            # NA_list=['None','<NA>']
            
            # from  matplotlib.colors import LinearSegmentedColormap
            # cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
            # cm=sns.dark_palette("#282d", reverse=True, as_cmap=True)#169
            # cmi=sns.dark_palette("#282d", reverse=False, as_cmap=True)# 
            # cm=sns.color_palette("vlag_r", as_cmap=True)
            # cm=sns.color_palette("dark:salmon_r", as_cmap=True)
            # cm=sns.palplot(sns.dark_palette("seagreen", reverse=True))
            # cm=sns.palplot(sns.dark_palette((260, 75, 60), input="husl"))
            # cm=sns.diverging_palette(20, 150, as_cmap=True) 
            # cmi=sns.diverging_palette(150, 20, as_cmap=True) 
            cm=sns.diverging_palette(150,20,l=35,center="dark",  as_cmap=True) 
            cmi=sns.diverging_palette(20,150,l=35,center="dark",  as_cmap=True) 
            
            # cm=sns.color_palette("icefire", as_cmap=True)
            
            df_fond_price=(df_fondamental_analyse[Fond_metrics_list_price]
            .replace('None','NaN')
            .rename(index=dict_ticker_to_name)
            .sort_index()
            .astype(float)
            .style
            .format("{:.2f}", na_rep="-")
            .background_gradient(axis=0,cmap=cm))#,low=0.3, high=0.05)
            
            df_fond_profitability=(df_fondamental_analyse[Fond_metrics_list_profitability]
            .replace('None','NaN')
            .rename(index=dict_ticker_to_name)
            .sort_index()
            .astype(float)
            .style
            .format("{:.2%}", na_rep="-")
            .format({'revenuePerShare': '{:,.2f}'.format})
            .background_gradient(axis=0,cmap=cmi))#,low=0, high=0)
            
            df_fond_BS=(df_fondamental_analyse[Fond_metrics_list_BS]
            .replace('None','NaN')
            .rename(index=dict_ticker_to_name)
            .sort_index()
            .astype(float)
            .style.format('{:.2f}', na_rep="-")
            .format({
                'returnOnAssets': '{:,.2%}'.format,
                'returnOnEquity': '{:,.2%}'.format,
                })
            .background_gradient(axis=0,cmap=cmi))#,low=0.3, high=0.05)
            # , subset=['Temp (c)', 'Rain (mm)', 'Wind (m/s)']
            
            df_fond_Market=(df_fondamental_analyse[Fond_metrics_list_Market]
            .replace({'None':'NaN',0:'NaN'})
            .rename(index=dict_ticker_to_name)
            .sort_index()
            .astype(float)
            .style
            .format('{:.2f}', na_rep="-")
            .background_gradient(axis=0,cmap=cm))#,low=0.3, high=0.05)
            
            df_fond_Absolute=(df_fondamental_analyse[Fond_metrics_list_Absolute]
            .replace('None','NaN')
            .rename(index=dict_ticker_to_name)
            .sort_index()
            .astype(float)
            .style
            .format('{:.2f}', na_rep="-")
            .background_gradient(axis=0,cmap=cm))#,low=0.3, high=0.05)
            # df_fondamental_analyse[Fond_metrics_list_price].style.background_gradient()
            # st.write(df_fondamental_analyse[Fond_metrics_list_BS].loc['ACA.PA','debtToEquity'])
            # st.write(df_fondamental_analyse[Fond_metrics_list_price].loc['ACA.PA','pegRatio'])
            # df_fond_price
            st.subheader("Fondamental Data")
            
            st.caption("Ratio - Price")
            st.dataframe(df_fond_price,width=1500,height=2000)
            
            st.caption("Ratio - Profitability")
            st.dataframe(df_fond_profitability,width=1500,height=2000)
            
            st.caption("Ratio - Balance Sheet")
            st.dataframe(df_fond_BS,width=1500,height=2000)
            
            st.caption("Balance Sheet data")
            st.dataframe(df_fond_Absolute,width=1500,height=2000)            
            
            st.subheader("Market Data")
            
            st.caption("Ratio - Market data")
            st.dataframe(df_fond_Market,width=1500,height=2000)
            # df_fondamental_Index[df_fondamental_Index['Attribute']==Sector_name][df_fondamental_Index['Attribute']==Industry_name][df_fondamental_Index['Recent']==Industry].drop_duplicates().tolist()
            # for e in Sector_Ticker:
                # st.write(dict_ticker_to_name[e])
            
            # st.write(Sector_Ticker)
            
            #Retraitement des doublons dans plusieurs indeices
            df_prices_analyse=df_prices.copy()
            df_prices_analyse.columns=df_prices_analyse.columns.droplevel(0)
            # st.dataframe(df_prices_analyse.columns)
            # st.dataframe(df_prices_analyse)
            df_prices_analyse=df_prices_analyse[df_fondamental_Index_filter_ticker]
            df_prices_analyse=df_prices_analyse.loc[:,~df_prices_analyse.columns.get_level_values(0).duplicated(keep='first')]
            # st.write(df_prices_analyse.columns.get_level_values(0))
            # st.write(list(df_prices_analyse.columns.get_level_values(0).duplicated(keep='first')))
            # st.dataframe(df_prices_analyse)
            #Dataframe des data de perf
            df_prices_ytd=df_prices_analyse.copy()
            df_prices_ytd['year']=df_prices_ytd.index.year#.rename('year')
            
            df_prices_perf=(df_prices_ytd
            .groupby('year')
            .last()
            .pct_change())
            
            # st.dataframe(df_prices_perf)
            
            df_prices_ytd_first=(df_prices_ytd
            .groupby('year')
            .first()).loc[datetime.today().year,:]

            df_prices_ytd_last=(df_prices_ytd
            .groupby('year')
            .last()).loc[datetime.today().year,:]
            
            df_prices_ytd_perf=pd.DataFrame()
            
            df_prices_ytd_perf['col1']=df_prices_ytd_first
            df_prices_ytd_perf['col2']=df_prices_ytd_last
            df_prices_ytd_perf['YTD']=df_prices_ytd_perf.pct_change(axis=1)['col2']
            
            df_prices_ytd_perf=df_prices_ytd_perf[['YTD']]
            
            df_prices_perf=(df_prices_perf[df_prices_perf.index>datetime.today().year-5]       
            .T)
            
            df_prices_perf[datetime.today().year]=df_prices_ytd_perf
            
            # st.dataframe(df_prices_perf.index,width=1500,height=2000)
            
            df_prices_perf=(df_prices_perf
            .rename(index=dict_ticker_to_name)
            .style
            .format('{:.2%}', na_rep="-")
            .background_gradient(axis=0,cmap=cmi))
            
            st.caption("Perf Histo")
            st.dataframe(df_prices_perf,width=1500,height=2000)
            
            st.line_chart(df_prices_analyse.rename(columns=dict_ticker_to_name).dropna())
                       
            # st.write(type(pf.create_simple_tear_sheet(df_prices['ACA.PA'])))
            
            # for e in list(df_prices.columns):
            #     data=df_prices[e]
                # st.write(type(pf.create_simple_tear_sheet(data)))
            # st.write(Sector_list)
    
elif page == "Focus":
    
    df_market_Focus=df_market.copy()
    df_market_Focus.columns=df_market.columns.droplevel(0)
       
    df_prices.columns=df_prices.columns.droplevel(0)
    # st.write(df_prices)
        
    
    # st.caption(stock_name_focus)
    # st.write(df_prices[dict_name_to_ticker[stock_name_Index]])
    # st.dataframe(df_market_Focus)

    # df_market_Focus_ticker=df_market_Focus_ticker.droplevel(0)
    
    if Index:
        stock_name_Focus=st.sidebar.selectbox("Select a stock", sorted(list(dict_ticker_to_name.values())))#list(df_prices.columns.get_level_values(1).unique()))
        if Index!=['Cryptocurrencies']:
                # stock_name=st.selectbox('Select a stock', sorted(dict_name_to_ticker.keys()))
        # st.line_chart(df_prices[dict_name_to_ticker[stock_name_focus]])    
    # st.write(df_prices[dict_name_to_ticker[stock_name_Index]])
    # pf.create_returns_tear_sheet()
            df_fondamental_Index_filter_ticker=df_fondamental_Index[df_fondamental_Index.Ticker.isin([dict_name_to_ticker[stock_name_Focus]])]
            st.dataframe(df_fondamental_Index_filter_ticker[df_fondamental_Index_filter_ticker.Attribute.isin(['industry','sector'])])
        df_prices_Focus=df_prices[dict_name_to_ticker[stock_name_Focus]].dropna()
        st.line_chart(df_prices_Focus)
        df_market_Focus_ticker=df_market_Focus.loc[:,df_market_Focus.columns.get_level_values(0)==dict_name_to_ticker[stock_name_Focus]]
        df_market_Focus_ticker.columns=df_market_Focus_ticker.columns.droplevel(0)
        df_market_Focus_ticker=df_market_Focus_ticker.dropna()
        
        #plot Ichimoku
        # df_market_Focus=df_market.loc[:, df_market.columns.get_level_values(0).isin(Index)]
        # st.dataframe(df_market_focus)
        # st.dataframe(df_market_focus.loc[:,df_market_focus.columns.get_level_values(0)==stock_name_focus])
        
        # st.dataframe(df_market_Focus_ticker)
        #############################################
        #Fill unique
        
        # df1=df_market_Focus_ticker.ta.ichimoku()[0]
        # df1['label'] = np.where(df1.ISA_9>df1.ISB_26, 1, 0)

        # def fillcol(label):
        #     if label == 1:
        #         return 'rgba(0,250,0,0.4)'
        #     else:
        #         return 'rgba(250,0,0,0.4)'
        # try:
        #     fig = px.line(df1[['ITS_9','IKS_26','ICS_26']],width=1300, height=600)
        #     fig.add_traces(go.Scatter(x=df1.index, y = df1.ISA_9,name="ISA_9"))
            
        #     fig.add_traces(go.Scatter(x=df1.index, y = df1.ISB_26,
        #                               name="ISB_26",
        #                               fill='tonexty', 
        #                               fillcolor = fillcol(df1['label'].iloc[0])))#'rgba(0,250,0,0.4)'))
        #     st.plotly_chart(fig)
        # except:
        #     st.write("no data for ichimaku")
        #############################################
        #Bon Fill
        
        df_ichi=df_market_Focus_ticker.ta.ichimoku()[0]
        # df_market_Focus_ticker.index = df_market_Focus_ticker.index.strftime("%d %b, %Y")
        # xaxis=dict(type = "category")
        # st.dataframe(df_ichi)
        df1=df_ichi[["ISA_9","ISB_26"]]
        # df1=df1.dropna()
        df1['label'] = np.where(df1.ISA_9>df1.ISB_26, 1, 0)
        df1['group'] = df1['label'].ne(df1['label'].shift()).cumsum()
        df1 = df1.groupby('group')
        # st.dataframe(df1)
        
        dfs = []
        for name, data in df1:
            dfs.append(data)
        # st.dataframe(dfs)
        # custom function to set fill color
        
        # try:
        # fig = px.line(df1[['ITS_9','IKS_26','ICS_26']],width=1300, height=600)
        layout = go.Layout(xaxis_rangeslider_visible=False,width=1500,height=600)
        # fig.update_layout(xaxis_rangeslider_visible=False
        fig = go.Figure(layout=layout) 
        
        fig.add_traces(go.Candlestick(x=df_market_Focus_ticker.index,
                                    open=df_market_Focus_ticker['Open'],
                                    high=df_market_Focus_ticker['High'],
                                    low=df_market_Focus_ticker['Low'],
                                    close=df_market_Focus_ticker['Close'],name="Stock"))
        fig.add_traces(go.Scatter(x=df_ichi.index,
                                  y = df_ichi.ITS_9,
                                  line = dict(color = '#ff7f0e', width=1),
                                  name="ITS_9"))
        fig.add_traces(go.Scatter(x=df_ichi.index,
                                  y = df_ichi.ICS_26,
                                  line = dict(color = 'brown', width=1),
                                  name="ICS_26"))
        fig.add_traces(go.Scatter(x=df_ichi.index,
                                  y = df_ichi.IKS_26,
                                  line = dict(color = '#1f77b4', width=1),
                                  name="IKS_26"))
        for df in dfs:
            fig.add_traces(go.Scatter(x=df.index,
                                      y = df.ISA_9,
                                      showlegend=False,
                                      line = dict(color='#2ca02c', width=0.5)))
            
            fig.add_traces(go.Scatter(x=df.index,
                                      y = df.ISB_26,
                                      fill='tonexty', 
                                      fillcolor = fillcol(df['label'].iloc[0]),
                                      showlegend=False,
                                      line = dict(color='#d62728', width=0.5)))#'rgba(0,250,0,0.4)'))
            
        Fig_candlestick_setup(fig)
        #############################################
        
        # a1 = df1.plot()
        # a1.fill_between(df1.index, df1.ISA_9, df1.ISB_26)
        st.plotly_chart(fig)

        #############################################
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=df1.index, y=df1[['ISA_9','ITS_9','IKS_26','ICS_26']],
        #     fill=None,
        #     mode='lines',
        #     line_color='indigo',
        #     ))
        # fig.add_trace(go.Scatter(
        #     x=df1.index,
        #     y=df1[['ISB_26']],
        #     fill='tonexty', # fill area between trace0 and trace1
        #     mode='lines', line_color='indigo'))
        
        # def fillcol(label):
        #     if label >= 1:
        #         return 'rgba(0,250,0,0.4)'
        #     else:
        #         return 'rgba(250,0,0,0.4)'
            
        # df1['label'] = np.where(df1['ISA_9']>df1['ISB_26'], 1, 0)
        
        # st.dataframe(df1)
        

        
        df2=ta.bbands(df_market_Focus_ticker['Close']).dropna()
        # st.dataframe(df_market_Focus_ticker)
        # st.dataframe(df2)
        # fig2 = px.line(df_market_Focus_ticker['Close'],width=1300, height=800)
        fig2 = px.line(df2[['BBL_5_2.0','BBU_5_2.0']],width=1350, height=600)#'BBM_5_2.0'
        fig2.add_traces(go.Scatter(x=df_market_Focus_ticker.index, y = df_market_Focus_ticker['Close'],name="Stock Close Price"))
        st.plotly_chart(fig2)
        
        df3=ta.macd(df_market_Focus_ticker['Close']).dropna()
        df4=ta.rsi(df_market_Focus_ticker['Close']).to_frame().dropna()
        # st.dataframe(df3)
        # st.dataframe(df4)
        # st.write(df4.columns)
        fig3 = make_subplots(rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.01)
        # fig2 = px.line(df_market_Focus_ticker['Close'],width=1300, height=800)
        # fig3 = px.line(df3[['MACD_12_26_9','MACDh_12_26_9']],width=1300, height=800)
        fig3.append_trace(go.Scatter(x=df_market_Focus_ticker.index, y = df_market_Focus_ticker['Close'],name="Stock Price"), row=1, col=1)
        fig3.append_trace(go.Scatter(x=df3.index, y = df3['MACD_12_26_9'],name="MACD"), row=2, col=1)
        fig3.append_trace(go.Scatter(x=df4.index, y = df4["RSI_14"],name="RSI"), row=3, col=1)
        # fig3.append_trace(go.Scatter(x=df2.index, y = df2['BBL_5_2.0'],name="BBANDS1"), row=3, col=1)
        # fig3.add_traces(go.Scatter(x=df2.index, y = df2['BBU_5_2.0'],name="BBANDS"))
        fig3.update_layout(height=1000, width=1300, title_text="Momentum indicators")
        st.plotly_chart(fig3)
    
    # st.write(df_fondamental_Index[df_fondamental_Index.Ticker==dict_name_to_ticker[stock_name_Index]])

elif page == "Technical Analysis":
    
    TA_analysis_type=st.sidebar.selectbox("TA Analysis :", ["All Market","Stock Focus"])
    
    df_market_TA=df_market.copy()
        
    if TA_analysis_type=="All Market":
        patterns_sorted.append("All")
        pattern_list = st.multiselect("Choose your technical analysis", patterns_sorted)
        if "All" in pattern_list:
            pattern_list = list(candlestick_patterns.values())
        else:
            pattern_list = pattern_list
    # Index_sorted_focus=Index_sorted.append('All')
        
        for pattern in pattern_list:
            for ticker in sorted(df_market_TA.columns.get_level_values(1).unique().tolist()):
                pattern_function = getattr(talib, list(candlestick_patterns.keys())[list(candlestick_patterns.values()).index(pattern)])
                
                # try:
                # st.write(ticker)
                # st.write("begin loop")
                # st.write(df_market_TA)
                df_market_TA_ticker=df_market_TA.loc[:,df_market_TA.columns.get_level_values(1)==ticker]
                # st.write(df_market_TA)
                # df_market_TA_ticker=df_market_TA.copy()
                df_market_TA_ticker.columns=df_market_TA_ticker.columns.droplevel(0)
                df_market_TA_ticker.columns=df_market_TA_ticker.columns.droplevel(0)
                df_market_TA_ticker=df_market_TA_ticker.dropna()
                # st.write(df_market_TA_ticker)
                results = pattern_function(df_market_TA_ticker['Open'], df_market_TA_ticker['High'], df_market_TA_ticker['Low'], df_market_TA_ticker['Close'])
                last = results.tail(1).values[0]
                if last > 0:
                    ind = 'bullish'
                elif last < 0:
                    ind = 'bearish'
                else:
                    ind = None
                
                if ticker in dict_ticker_to_name:
                    ticker_name=dict_ticker_to_name[ticker]
                else:
                    ticker_name=ticker
                
                if ind!=None:                    
                            fig = go.Figure(data=[go.Candlestick(x=df_market_TA.index,
                                            open=df_market_TA_ticker['Open'],
                                            high=df_market_TA_ticker['High'],
                                            low=df_market_TA_ticker['Low'],
                                            close=df_market_TA_ticker['Close'])])
                            
                            fig.update_layout(
                                title=ticker_name+' - '+ind ,
                                xaxis_title="Date",
                                yaxis_title="Price",
                            )
                            fig.update_layout(xaxis_rangeslider_visible=False,width=1300,height=600)
                            # fig.data[1].decreasing.line.color = 'rgba(0,0,0,0)'
                            st.plotly_chart(fig)
    elif TA_analysis_type=="Stock Focus":
        stock_list=sorted(list(dict_ticker_to_name.values()))
        Stock_TA=st.sidebar.selectbox("Select a stock :", stock_list)
        if Stock_TA!=None:
            Stock_TA_tick=dict_name_to_ticker[Stock_TA]
            i=0
        
            for pattern in patterns_sorted:
                pattern_function = getattr(talib, list(candlestick_patterns.keys())[list(candlestick_patterns.values()).index(pattern)])
                # st.write(Stock_TA_tick)
                # try:
                # st.write(ticker)
                # st.write("begin loop")
                # st.dataframe(df_market_TA)
                # st.dataframe(df_market_TA_ticker)
                df_market_TA_ticker=df_market_TA.loc[:,df_market_TA.columns.get_level_values(1)==Stock_TA_tick]
                # st.write(df_market_TA)
                # df_market_TA_ticker=df_market_TA.copy()
                # st.dataframe(df_market_TA_ticker)
                df_market_TA_ticker.columns=df_market_TA_ticker.columns.droplevel(0)
                df_market_TA_ticker.columns=df_market_TA_ticker.columns.droplevel(0)
                df_market_TA_ticker=df_market_TA_ticker.dropna()
                
                # st.dataframe(df_market_TA_ticker)
                # st.write(pattern)
                results = pattern_function(df_market_TA_ticker['Open'], df_market_TA_ticker['High'], df_market_TA_ticker['Low'], df_market_TA_ticker['Close'])
                last = results.tail(1).values[0]
                if last > 0:
                    ind = 'bullish'
                elif last < 0:
                    ind = 'bearish'
                else:
                    ind = None
                
                if Stock_TA in dict_ticker_to_name:
                    ticker_name=dict_ticker_to_name[Stock_TA]
                else:
                    ticker_name=Stock_TA
                
                if ind!=None:                    
                    fig = go.Figure(data=[go.Candlestick(opacity=1,x=df_market_TA.index,
                                    open=df_market_TA_ticker['Open'],
                                    high=df_market_TA_ticker['High'],
                                    low=df_market_TA_ticker['Low'],
                                    close=df_market_TA_ticker['Close']
                                    )])
                    
                    fig.update_layout(
                        title=ticker_name+' - '+ pattern +' - '+ind ,
                        xaxis_title="Date",
                        width=1500,height=600,
                        yaxis_title="Price"
                    )
                    
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    
                    Fig_candlestick_setup(fig)
                    
                    st.plotly_chart(fig)
                    ############################################################################
                    #Altair test
                    # base = alt.Chart(df_market_TA).encode(
                    #     color=alt.condition("datum.Open <= datum.Close",
                    #                         alt.value("#06982d"), alt.value("#ae1325"))
                    # )
                    
                    # rule = base.mark_rule().encode(alt.Y('Low:Q', title='Price',
                    #                                         scale=alt.Scale(zero=False)), alt.Y2('High:Q'))
                    # bar = base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q'))
                    # st.altair_chart(rule + bar, use_container_width=True)
                    ############################################################################
                else:    
                   i+=1
            if len(patterns_sorted)==i:
                st.write("No Technical indicators bullish or bearish for "+str(ticker_name)+" stock")
# Index=['NASDAQ']


# test=list(dict_ticker_to_name.values())
###############################################
#Sector definition

# stock_list=list(dict_ticker_to_name.values())#.insert(0, "All")
# Stock_TA=st.sidebar.selectbox("Select a stock :", stock_list)

print("Temps d'execution : %s secondes" % (time.time() - start_time))