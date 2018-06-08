#-*-coding:utf-8-*-
import pandas as pd
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import savefig
#1.读数据
def read(str):
    data=pd.read_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv", encoding="gbk")
    return data
#2.写数据
def write(data,str):
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv", header=None)
#3.resource
def predict():
    # ------------------------------------week1
    week = 1
    data = read("data_preprocessing\\fillna_week_" + str(week))
    i_1=0.1
    j_1=0.1
    k_1=0.32
    s_1=0.48
    data['20161107']=data.ix[:,'20161010']*i_1+data.ix[:,'20161017']*j_1\
                     +data.ix[:,'20161024']*k_1+data.ix[:,'20161031']*s_1
    data['20161114']=data['20161107']
    pre_week1=data[['shop_id','20161107','20161114']]

    #------------------------------------week2
    week=2
    data2 = read("data_preprocessing\\fillna_week_" + str(week))
    i_2 = 0.0
    j_2 = 0.2
    k_2 = 0.32
    s_2 = 0.48
    data2['20161101'] = data2.ix[:, '20160927'] * i_2 + data2.ix[:, '20161011'] * j_2 \
                       + data2.ix[:, '20161018'] * k_2 + data2.ix[:, '20161025'] * s_2
    data2['20161108'] = data2['20161101']
    pre_week2=data2[['shop_id','20161101','20161108']]

    # ------------------------------------week3
    week = 3
    data3 = read("data_preprocessing\\fillna_week_" + str(week))
    i_3 = 0.0
    j_3 = 0.2
    k_3 = 0.32
    s_3 = 0.48
    data3['20161102'] = data3.ix[:, '20160928'] * i_3 + data3.ix[:, '20161012'] * j_3 \
                        + data3.ix[:, '20161019'] * k_3 + data3.ix[:, '20161026'] * s_3
    data3['20161109'] = data3['20161102']
    pre_week3 = data3[['shop_id', '20161102', '20161109']]

    # ------------------------------------week4
    week = 4
    data4 = read("data_preprocessing\\fillna_week_" + str(week))
    i_4 = 0.0
    j_4 = 0.2
    k_4 = 0.32
    s_4 = 0.48
    data4['20161103'] = data4.ix[:, '20160929'] * i_4 + data4.ix[:, '20161013'] * j_4 \
                        + data4.ix[:, '20161020'] * k_4 + data4.ix[:, '20161027'] * s_4
    data4['20161110'] = data4['20161103']
    pre_week4 = data4[['shop_id', '20161103', '20161110']]

    # ------------------------------------week5
    week = 5
    data5 = read("data_preprocessing\\fillna_week_" + str(week))
    i_5 = 0.0
    j_5 = 0.2
    k_5 = 0.3
    s_5 = 0.5
    data5['20161104'] = data5.ix[:, '20160923'] * i_5 + data5.ix[:, '20161014'] * j_5 \
                        + data5.ix[:, '20161021'] * k_5 + data5.ix[:, '20161028'] * s_5
    data5['20161111'] = data5['20161104'] * 1.2
    pre_week5 = data5[['shop_id', '20161104', '20161111']]

    # ------------------------------------week6
    week = 6
    data6 = read("data_preprocessing\\fillna_week_" + str(week))
    i_6 = 0.05
    j_6 = 0.15
    k_6 = 0.32
    s_6 = 0.48
    data6['20161105'] = data6.ix[:, '20160924'] * i_6 + data6.ix[:, '20161015'] * j_6 \
                        + data6.ix[:, '20161022'] * k_6 + data6.ix[:, '20161029'] * s_6
    data6['20161112'] = data6['20161105']
    pre_week6 = data6[['shop_id', '20161105', '20161112']]

    # ------------------------------------week7
    week = 7
    data7 = read("data_preprocessing\\fillna_week_" + str(week))
    i_7 = 0.0
    j_7 = 0.2
    k_7 = 0.32
    s_7 = 0.48
    data7['20161106'] = data7.ix[:, '20160925'] * i_7 + data7.ix[:, '20161016'] * j_7 \
                        + data7.ix[:, '20161023'] * k_7 + data7.ix[:, '20161030'] * s_7
    data7['20161113'] = data7['20161106']
    pre_week7 = data7[['shop_id', '20161106', '20161113']]

    prediction=read("Predict_Result\\model")

    prediction = pd.merge(prediction, pre_week1, on='shop_id')
    prediction = pd.merge(prediction, pre_week2, on='shop_id')
    prediction = pd.merge(prediction, pre_week3, on='shop_id')
    prediction = pd.merge(prediction, pre_week4, on='shop_id')
    prediction = pd.merge(prediction, pre_week5, on='shop_id')
    prediction = pd.merge(prediction, pre_week6, on='shop_id')
    prediction = pd.merge(prediction, pre_week7, on='shop_id')




    prediction=prediction[['shop_id','20161101','20161102','20161103','20161104','20161105','20161106','20161107'
                      ,'20161108','20161109','20161110','20161111','20161112','20161113','20161114']]

    print prediction

    for i in xrange(prediction.shape[1]):
        prediction.ix[:,i]=prediction.ix[:,i].apply(lambda x:round(x))
    print prediction
    write(prediction,"Predict_Result\\0208\\prediction1")
def Write_Predict_Model(data,strs):
    prediction = read("Predict_Result\\model")
    prediction = pd.merge(prediction, data, on='shop_id')
    prediction = prediction[
        ['shop_id', '20161101', '20161102', '20161103', '20161104', '20161105', '20161106', '20161107'
            , '20161108', '20161109', '20161110', '20161111', '20161112', '20161113', '20161114']]
    for i in xrange(prediction.shape[1]):
        prediction.ix[:, i] = prediction.ix[:, i].apply(lambda x: round(x))
    prediction = prediction.set_index('shop_id')
    write(prediction, "Predict_Result\\"+strs)


#---week1--1.提交0210_
def predict1_type1():
    predict_data=read("Predict_Result\\0215\\predict2")

    week=1
    # type=1
    data=read("cluster_test_everyweekday\\cluster_week"+str(week))
    # data= data[data['type']==type]
    # print data

    data['a'] = data.ix[:,'20160912']* 0.1\
                + data.ix[:,'20160919']* 0.01\
                + data.ix[:, '20160926'] *0.1\
                + data.ix[:, '20161010'] * 0.18\
                + data.ix[:, '20161017'] * 0.1 \
                + data.ix[:, '20161024'] * 0.01\
                + data.ix[:, '20161031'] * 0.5

    predicts=pd.merge(predict_data,data[['shop_id','a']],how='left',on='shop_id')

    pre1=predicts[predicts['a'].isnull()==True]
    pre2=predicts[predicts['a'].isnull()==False]

    pre2['20161107'] = pre2['a']
    pre2['20161114'] = pre2['a']

    data=pd.concat([pre1,pre2])
    print data[['20161107','20161114','a']]
    Write_Predict_Model(data, "0215\\predict3")
#week1_type_0
def predict1_type0():
    predict_data=read("Predict_Result\\0302\\shev1")

    week=1
    type=6
    data=read("cluster_test_everyweekday\\cluster_week"+str(week))
    data= data[data['type']==type]

    data['a'] = data.ix[:, '20160912'] * 0.0 \
                + data.ix[:, '20160919'] * 0.0 \
                + data.ix[:, '20160926'] * 0.0 \
                + data.ix[:, '20161010'] * 0.0 \
                + data.ix[:, '20161017'] * 0.4 \
                + data.ix[:, '20161024'] * 0.0 \
                + data.ix[:, '20161031'] * 0.6

    predicts=pd.merge(predict_data,data[['shop_id','a']],how='left',on='shop_id')
    print predicts
    pre1=predicts[predicts['a'].isnull()==True]
    pre2=predicts[predicts['a'].isnull()==False]

    pre2['20161103'] = pre2['a']
    data=pd.concat([pre1,pre2])
    print data[['20161103','a']]
    Write_Predict_Model(data, "0303\\shev2")


predict1_type0()

def predict3_type0():
    predict_data=read("Predict_Result\\0222\\data")

    week=3
    type=0
    data=read("cluster_test_everyweekday\\cluster_week"+str(week))
    data= data[data['type']==type]

    data['a'] = data.ix[:, '20160914'] * 0.0 \
                + data.ix[:, '20160921'] * 0.0 \
                + data.ix[:, '20160928'] * 0.0 \
                + data.ix[:, '20161012'] * 0.1 \
                + data.ix[:, '20161019'] * 0.1 \
                + data.ix[:, '20161026'] * 0.8

    predicts=pd.merge(predict_data,data[['shop_id','a']],how='left',on='shop_id')
    print predicts
    pre1=predicts[predicts['a'].isnull()==True]
    pre2=predicts[predicts['a'].isnull()==False]

    pre2['20161102'] = pre2['a']
    data=pd.concat([pre1,pre2])
    print data[['20161102','a']]
    Write_Predict_Model(data, "0222\\data_2")

def predict4_type0():
    predict_data=read("Predict_Result\\0227\\gege_3")

    week=4
    type=0
    data=read("cluster_test_everyweekday\\cluster_week"+str(week))
    data= data[data['type']==type]

    data['a'] = data.ix[:, '20160908'] * 0.0 \
                + data.ix[:, '20160915'] * 0.0 \
                + data.ix[:, '20160922'] * 0.0 \
                + data.ix[:, '20161013'] * 0.1 \
                + data.ix[:, '20161020'] * 0.0 \
                + data.ix[:, '20161027'] * 0.9

    predicts=pd.merge(predict_data,data[['shop_id','a']],how='left',on='shop_id')
    print predicts
    pre1=predicts[predicts['a'].isnull()==True]
    pre2=predicts[predicts['a'].isnull()==False]

    pre2['20161103'] = pre2['a']
    data=pd.concat([pre1,pre2])
    print data[['20161103','a']]
    Write_Predict_Model(data, "0228\\g1")

#1.f:调整week_5的type_1
def predict5_type1():
    predict_data=read("Predict_Result\\0210\\f1_0210")

    week=5
    type=1
    data=read("cluster_test_everyweekday\\cluster_week"+str(week))
    data= data[data['type']==type]

    data['a'] = data.ix[:, '20160909'] * 0.0 \
                + data.ix[:, '20160916'] * 0.0 \
                + data.ix[:, '20160923'] * 0.0 \
                + data.ix[:, '20161014'] * 0.28 \
                + data.ix[:, '20161021'] * 0.22 \
                + data.ix[:, '20161028'] * 0.5

    predicts=pd.merge(predict_data,data[['shop_id','a']],how='left',on='shop_id')
    print predicts
    pre1=predicts[predicts['a'].isnull()==True]
    pre2=predicts[predicts['a'].isnull()==False]

    pre2['20161104'] = pre2['a']
    data=pd.concat([pre1,pre2])
    print data[['20161104','a']]
    Write_Predict_Model(data, "0211\\f1_w5_1")

#3.t:测week6的type1
def predict6_type1():
    predict_data = read("Predict_Result\\0210\\t3_w6_1")

    week = 6
    type = 1
    data = read("cluster_test_everyweekday\\cluster_week" + str(week))
    data = data[data['type'] == type]

    data['a'] = data.ix[:, '20160910'] * 0.0 \
                + data.ix[:, '20160917'] * 0.0 \
                + data.ix[:, '20160924'] * 0.0 \
                + data.ix[:, '20161015'] * 0.2 \
                + data.ix[:, '20161022'] * 0.48 \
                + data.ix[:, '20161029'] * 0.32

    predicts = pd.merge(predict_data, data[['shop_id', 'a']], how='left', on='shop_id')
    print predicts
    pre1 = predicts[predicts['a'].isnull() == True]
    pre2 = predicts[predicts['a'].isnull() == False]

    pre2['20161105'] = pre2['a']
    data = pd.concat([pre1, pre2])
    print data[['20161105', 'a']]
    Write_Predict_Model(data, "0211\\t2_w6_1")

#---shev--week7
def predict7_type1():
    predict_data=read("Predict_Result\\0210\\predict1_0210")

    week=7
    type=1
    data=read("cluster_test_everyweekday\\cluster_week"+str(week))
    data= data[data['type']==type]

    data['a'] = data.ix[:, '20160911'] * 0.00 \
                + data.ix[:, '20160918'] * 0.00 \
                + data.ix[:, '20160925'] * 0.00 \
                + data.ix[:, '20161016'] * 0.15 \
                + data.ix[:, '20161023'] * 0.55 \
                + data.ix[:, '20161030'] * 0.3

    predicts=pd.merge(predict_data,data[['shop_id','a']],how='left',on='shop_id')
    print predicts
    pre1=predicts[predicts['a'].isnull()==True]
    pre2=predicts[predicts['a'].isnull()==False]

    pre2['20161106'] = pre2['a']
    data=pd.concat([pre1,pre2])
    print data[['20161106','a']]
    Write_Predict_Model(data, "0211\\predict1_w_7_1")
