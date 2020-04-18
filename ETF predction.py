#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import talib
import xgboost as xgb

from sklearn import model_selection
from talib import abstract


# In[4]:


df_P=pd.read_csv('ETF NAV.txt',sep='\t',engine='python')


# In[5]:


df_P = df_P[['證券代碼', '簡稱', '年月日', '開盤價', '最高價', '最低價', '收盤價', '成交量', '成交值', '報酬率',
       '週轉率', '流通在外股數', '市值','本益比-TSE','現金股利率']]
df_P.iloc[:,3:len(df_P.columns)] = df_P.iloc[:,3:len(df_P.columns)].replace(' ', '', regex=True).replace('-', np.NaN, regex=True).replace('', np.NaN, regex=True)
                                                #去除空值以及-，把NONE轉為nan
df_P=df_P.dropna(axis=1,how='all').fillna(0)    #去除全部都是NA值的欄位
df_P.iloc[:,3:len(df_P.columns)] = df_P.iloc[:,3:len(df_P.columns)].apply(lambda x :x.astype('float')) 
df_P=df_P.sort_values(['證券代碼','年月日']).reset_index(drop=True)


# # 計算技術指標

# In[6]:


data = df_P[df_P['證券代碼']=='0050   ']
OHLCV_columns = [ '開盤價','最高價','最低價','收盤價','成交量' ]


# In[7]:


def calculator_talib(data):
    ETF = {
    'open':data[OHLCV_columns[0]].dropna().astype(float),
    'high':data[OHLCV_columns[1]].dropna().astype(float),
    'low':data[OHLCV_columns[2]].dropna().astype(float) ,
    'close':data[OHLCV_columns[3]].dropna().astype(float),
    'volume': data[OHLCV_columns[4]].dropna().astype(float)

    }

    def talib2df(talib_output):
        if type(talib_output) == list:
            ret = pd.DataFrame(talib_output).transpose()
        else:
            ret = pd.Series(talib_output)
        ret.index = data['收盤價'].index
        return ret

    KD = talib2df(abstract.STOCH(ETF, fastk_period=9))
#計算MACD#
    MACD = talib2df(abstract.MACD(ETF))
#計算OBV#
    OBV = talib2df(abstract.OBV(ETF))
#計算威廉指數#
    WILLR = talib2df(abstract.WILLR(ETF))
#ATR 計算#
    ATR = talib2df(abstract.ATR(ETF))
 


    ETF=pd.DataFrame()
    ETF = pd.concat([data,KD,MACD,OBV,WILLR,ATR], axis=1)
    return ETF


# In[8]:


df_P_Tech=df_P.groupby('證券代碼').apply(lambda x : calculator_talib(x))
df_P_Tech.columns = df_P.columns.tolist()+['k','d','dif12','dif26','macd','obv','willr','atr']
df_P_Tech = df_P_Tech.fillna(0)


# # 處理籌碼面資料

# In[9]:


df_Q=pd.read_csv('ETF Q.txt',sep='\t',engine='python')
df_Q = df_Q[[
             '證券代碼', '簡稱', '年月日',
             '外資買賣超(張)','投信買賣超(張)','自營買賣超(張)','外資買進金額(百萬)', '外資賣出金額(百萬)', '投信買進金額(百萬)', '投信賣出金額(百萬)','自營商買進金額(百萬)', '自營商賣出金額(百萬)', 
             '外資連續累計買賣超(張)', '投信連續累計買賣超(張)', '自營連續累計買賣超(張)','外資成交比重', '投信成交比重', '自營成交比重',
             '投信持股率％','自營持股率％','外資總持股數','投信持股數(張)','自營持股數',
             '信用交易比重','一般現股成交比重', '資券互抵(張)', '資券互抵比例',  
             '融資增加(張)', '融資減少(張)', '融券增加(張)', '融券減少(張)',
             '融資餘額(張)', '融資餘額(千元)', '融資使用率', '融券餘額(張)', '融券餘額(千元)', '融券使用率', '券資比'
            ]]
df_Q.iloc[:,3:len(df_Q.columns)] = df_Q.iloc[:,3:len(df_Q.columns)].replace(' ', '', regex=True).replace('-', np.NaN, regex=True).replace('', np.NaN, regex=True)
df_Q.iloc[:,3:len(df_Q.columns)] = df_Q.iloc[:,3:len(df_Q.columns)].apply(lambda x :x.astype('float'))
df_Q = df_Q.fillna(0)


# # 合併 創造Y欄位

# In[10]:


df_merge = pd.merge(left=df_P_Tech,right=df_Q,on=['證券代碼', '簡稱', '年月日'],how='left')
df_merge = df_merge.fillna(0).reset_index(drop=True)


# In[11]:


df_merge['Pt-5'] = df_merge.groupby('證券代碼')['收盤價'].shift(5) 
df_merge['Pt-4'] = df_merge.groupby('證券代碼')['收盤價'].shift(4) 
df_merge['Pt-3'] = df_merge.groupby('證券代碼')['收盤價'].shift(3) 
df_merge['Pt-2'] = df_merge.groupby('證券代碼')['收盤價'].shift(2) 
df_merge['Pt-1'] = df_merge.groupby('證券代碼')['收盤價'].shift(1) 
df_merge['diff7']= df_merge.groupby('證券代碼')['收盤價'].shift(-7) - df_merge['收盤價']


# # 回測XGBOOST 模型

# In[12]:


df_merge_dropna = df_merge.dropna()


# In[13]:


ETFlist = df_merge_dropna['證券代碼'].unique().tolist()

accuracy_list = []
train_length_list = []
test_length_list = []
for i in range(0,len(ETFlist)):

    temp = df_merge_dropna[df_merge_dropna['證券代碼']==ETFlist[i]]
    ETF_X = temp[['開盤價', '最高價', '最低價', '收盤價', '成交量', '成交值', '報酬率',
       '週轉率', '流通在外股數', '市值', '本益比-TSE', '現金股利率', 'k', 'd', 'dif12', 'dif26',
       'macd', 'obv', 'willr', 'atr', '外資買賣超(張)', '投信買賣超(張)', '自營買賣超(張)',
       '外資買進金額(百萬)', '外資賣出金額(百萬)', '投信買進金額(百萬)', '投信賣出金額(百萬)', '自營商買進金額(百萬)',
       '自營商賣出金額(百萬)', '外資連續累計買賣超(張)', '投信連續累計買賣超(張)', '自營連續累計買賣超(張)', '外資成交比重',
       '投信成交比重', '自營成交比重', '投信持股率％', '自營持股率％', '外資總持股數', '投信持股數(張)', '自營持股數',
       '信用交易比重', '一般現股成交比重', '資券互抵(張)', '資券互抵比例', '融資增加(張)', '融資減少(張)',
       '融券增加(張)', '融券減少(張)', '融資餘額(張)', '融資餘額(千元)', '融資使用率', '融券餘額(張)',
       '融券餘額(千元)', '融券使用率', '券資比','Pt-5', 'Pt-4', 'Pt-3', 'Pt-2', 'Pt-1']]
    ETF_y = (temp['diff7']>0)*1
    train_X, test_X, train_y, test_y = model_selection.train_test_split(ETF_X, ETF_y, test_size = 0.3)
    
    xgbc = xgb.XGBClassifier(n_estimators=200)
    xgbc.fit(train_X,train_y)
    accuracy_list.append( xgbc.score(test_X,test_y))
    train_length_list.append(len(train_X))
    test_length_list.append(len(test_X))
    print( xgbc.score(test_X,test_y) )
    


# In[14]:


df_accuracy = pd.DataFrame()
df_accuracy['ETF'] = ETFlist
df_accuracy['Accuracy'] = accuracy_list
df_accuracy['Train_data_length'] = train_length_list
df_accuracy['Test_data_length'] = test_length_list 


# In[15]:


df_accuracy.to_csv('All_ETF_Accuracy_n200.csv')


# # 迴圈做股價預測

# In[16]:


df_merge


# In[17]:


ETFlist = df_merge['證券代碼'].unique().tolist()
namelist = df_merge['簡稱'].unique().tolist()

predict_proba_list = []
predict_list = []

train_length_list = []
test_length_list = []
for i in range(0,len(ETFlist)):

    temp = df_merge[df_merge['證券代碼']==ETFlist[i]].reset_index(drop=True)
    temp_dropna = temp.dropna()
    
    ETF_X = temp_dropna[['開盤價', '最高價', '最低價', '收盤價', '成交量', '成交值', '報酬率',
       '週轉率', '流通在外股數', '市值', '本益比-TSE', '現金股利率', 'k', 'd', 'dif12', 'dif26',
       'macd', 'obv', 'willr', 'atr', '外資買賣超(張)', '投信買賣超(張)', '自營買賣超(張)',
       '外資買進金額(百萬)', '外資賣出金額(百萬)', '投信買進金額(百萬)', '投信賣出金額(百萬)', '自營商買進金額(百萬)',
       '自營商賣出金額(百萬)', '外資連續累計買賣超(張)', '投信連續累計買賣超(張)', '自營連續累計買賣超(張)', '外資成交比重',
       '投信成交比重', '自營成交比重', '投信持股率％', '自營持股率％', '外資總持股數', '投信持股數(張)', '自營持股數',
       '信用交易比重', '一般現股成交比重', '資券互抵(張)', '資券互抵比例', '融資增加(張)', '融資減少(張)',
       '融券增加(張)', '融券減少(張)', '融資餘額(張)', '融資餘額(千元)', '融資使用率', '融券餘額(張)',
       '融券餘額(千元)', '融券使用率', '券資比','Pt-5', 'Pt-4', 'Pt-3', 'Pt-2', 'Pt-1']]
    ETF_y = (temp_dropna['diff7']>0)*1
    
    train_X = ETF_X.iloc[0:(len(ETF_X)-1),:]
    train_Y = ETF_y[0:(len(ETF_X)-1)]
    
    
    test_X =  temp[train_X.columns.tolist()].iloc[len(temp)-1,:]
    test_X =  pd.DataFrame(test_X).T
    
    
    xgbc = xgb.XGBClassifier(n_estimators=200)
    
    predict_list.append( xgbc.fit(train_X,train_Y).predict(test_X)[0] )
    predict_proba_list.append( xgbc.fit(train_X,train_Y).predict_proba(test_X)[0][1])
    
    train_length_list.append(len(train_X))
    test_length_list.append(len(test_X))
    


# In[18]:


namelist = df_merge['簡稱'].unique().tolist()

df_predict = pd.DataFrame()
df_predict['ETF'] = ETFlist
df_predict['Name'] = namelist
df_predict['predict'] = predict_list
df_predict['predict_proba'] = predict_proba_list
df_predict['Train_data_length'] = train_length_list
df_predict['Test_data_length'] = test_length_list 


# In[19]:


df_predict.to_csv('df_Predict.csv')


# In[20]:


# 占整體模型 30%

ML_allocation = 0.3

df_predict = df_predict[df_predict.Train_data_length >= 500] #取資料行數大於500之資料
df_predict = df_predict[df_predict.predict == 1]             #7天后大於零的狀況
df_predict['Rank_1-20'] = df_predict['predict_proba'].rank(ascending=False) #最大數為1
df_predict = df_predict[df_predict['Rank_1-20'] <= 10]
df_predict['Rank_20-1'] = df_predict['predict_proba'].rank(ascending=True)  #最大數為20
df_predict['Weight'] = round(df_predict['Rank_20-1']/df_predict['Rank_20-1'].sum(),3) * ML_allocation


# In[21]:


df_predict.to_csv('df_Predict_final_selection.csv',index=False,encoding='utf-8')


# In[22]:


df_predict


# In[ ]:




