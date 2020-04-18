# XGboost-ETF

利用機器學習模型做資產預測，運用於ETF交易競賽。
競賽周期為2019/03/25~2019/06/03，投資標的為台灣上市所有ETF。


## Data Source : TEJ
https://drive.google.com/drive/folders/16Of9Wfffb50az0U87iHjmFLgzjbs5HTR

1. ETF NAV.txt : ETF淨值資料，及交易數據

   使用之資料欄位 : 

              '證券代碼', '簡稱', '年月日', '開盤價', '最高價', '最低價', '收盤價', 
              '成交量', '成交值', '報酬率', '週轉率', '流通在外股數', '市值',
              '本益比-TSE','現金股利率'

2. ETF Q.txt   : ETF籌碼面資訊

   使用之資料欄位 : 

             '證券代碼', '簡稱', '年月日',
             '外資買賣超(張)','投信買賣超(張)','自營買賣超(張)','外資買進金額(百萬)', '外資賣出金額(百萬)', '投信買進金額(百萬)', '投信賣出金額(百萬)','自營商買進金額(百萬)', '自營商賣出金額(百萬)', 
             '外資連續累計買賣超(張)', '投信連續累計買賣超(張)', '自營連續累計買賣超(張)','外資成交比重', '投信成交比重', '自營成交比重',
             '投信持股率％','自營持股率％','外資總持股數','投信持股數(張)','自營持股數',
             '信用交易比重','一般現股成交比重', '資券互抵(張)', '資券互抵比例',  
             '融資增加(張)', '融資減少(張)', '融券增加(張)', '融券減少(張)',
             '融資餘額(張)', '融資餘額(千元)', '融資使用率', '融券餘額(張)', '融券餘額(千元)', '融券使用率', '券資比'
             
             
## 計算技術指標

    以TA-Lib套件，計算以下指標，包含KD、MACD、OBV、WILLR、ATR等
    
    KD = talib2df(abstract.STOCH(ETF, fastk_period=9))
    MACD = talib2df(abstract.MACD(ETF))
    OBV = talib2df(abstract.OBV(ETF))
    WILLR = talib2df(abstract.WILLR(ETF))
    ATR = talib2df(abstract.ATR(ETF))

## 處理籌碼面資料

   整理資料以及填入缺值

## 合併 創造Y欄位

   將籌碼面及技術面資料合併，用以下欄位作為Key合併 : ['證券代碼', '簡稱', '年月日']
   
   Y為預測標的，將T+7日報酬>0的樣本標記為1

## 回測XGBOOST 模型

   以XGBOOST模型將資料切分為3分之2訓練集，3分之1測試集，觀察每檔ETF歷史模型預測準確率
   整體ETF預測準確率約在70%~80%左右

## 迴圈做股價預測

   以Expanding Window的方式預測，
   ![image](https://github.com/xavior107352022/XGboost-ETF-/blob/master/cummulativ_return.jpg)
   ![image](https://github.com/xavior107352022/XGboost-ETF-/blob/master/period_accuracy.jpg)
   ![image](https://github.com/xavior107352022/XGboost-ETF-/blob/master/statistic.jpg)
             
