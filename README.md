# XGboost-ETF

利用機器學習模型做資產預測，運用於ETF交易競賽。
競賽周期為2019/03/25~2019/06/03，投資標的為台灣上市所有ETF，每周換倉、計算績效。


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

   以Expanding Window的方式預測，用過去所有可取得資料預測T+7報酬漲跌，
   
   XGBOOST模型參數只更改n_estimators=200，輸出soft_max值，概念類似標的上漲之機率，再以此值作為權重加權。
   
   
   以下為期間準確率，約為0.53仍有進步空間
   
   ![image](https://github.com/xavior107352022/XGboost-ETF-/blob/master/period_accuracy.jpg)
   
   觀察每一期準確率，如下圖，我們可以發現，以黃色標示之日期為市場較為動盪的期間，
   
   模型準確率就會顯著降低，導致績效下降。
   
   ![image](https://github.com/xavior107352022/XGboost-ETF-/blob/master/statistic.jpg)
       
## ETF交易競賽期間損益

   競賽期間主觀交易累積報酬顯著高於大盤，而機器學習模型(XGBOOST)績效較差，可以觀察到模型在市場劇烈動盪時無法及時反應。

   ![image](https://github.com/xavior107352022/XGboost-ETF-/blob/master/cummulativ_return.jpg)
   
## 結論及檢討

   未來加入VIX、美元、黃金等資產衡量投資人恐慌狀態，或許能夠捕捉到市場動盪的狀態。
