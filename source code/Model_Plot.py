#-*-coding:utf-8-*-
import random

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
from datetime import timedelta
from scipy.stats import pearsonr


#1.读数据
def read(str):
    data=pd.read_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv", encoding="gbk")
    return data

#设置一个类似于全局的数据，免得每次都读，这样我只读一次
#注：Train_Data是date小于2016-10-18的数据，Vali_Data是date大于等于2016-10-18的数据
Train_Data = read("train\\Train_Data")
Vali_Data = read("vali\\Vali_Data")
mod_vali=Vali_Data[['shop_id','date','week']]#提交的模板

#2.写数据
def write(data,str):
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv", encoding="gbk")

#3.画出评测函数图像
def plotxy():
    x=np.arange(0,100,0.1)
    print x

    plt.plot(x,abs((x-50)/(x+50)))
#4.由于不同的weekday的起止日期不一样，所以该方法为方便得到起止日期
def beginend(weekday):
    if weekday==1:
        begin='20160905'
        end='20161017'
        end_vali='20161031'
    elif weekday == 2:
        begin = '20160830'
        end = '20161011'
        end_vali = '20161025'
    elif weekday == 3:
        begin = '20160831'
        end = '20161012'
        end_vali = '20161026'
    elif weekday == 4:
        begin = '20160901'
        end = '20161013'
        end_vali = '20161027'
    elif weekday == 5:
        begin = '20160902'
        end = '20161014'
        end_vali = '20161028'
    elif weekday == 6:
        begin = '20160903'
        end = '20161015'
        end_vali = '20161029'
    else:
        begin = '20160904'
        end = '20161016'
        end_vali = '20161030'
    return begin,end,end_vali

#5.对聚类后的数据画图
def plot_cluster(weekday):
    begin, end, end_vali = beginend(weekday)
    ticks=[0,1,2,3,4,5]
    # week=read("cluster_everyweekday\\cluster_1\\weekday"+str(weekday))              #没有验证两天的数据
    week=read("cluster_everyweekday\\cluster_1\\weekday_vali"+str(weekday))     #有验证两天的数据
    end=end_vali#有验证两天的数据后end也发生变化
    ticks=[0,1,2,3,4,5,6,7]
    for i in xrange(8):
        weeki=week[week['type']==i]
        weeki=weeki.reset_index(drop=True)
        print weeki
        df = weeki.ix[:,begin:end]
        plt.legend('x')
        # 筛选出100以内的,100至400的，大于400，好观察
        df1=df[(df[begin]<100) & (df[end]<100) ]
        df1 = df1.T
        df1.plot(kind='line', alpha=1, xticks=ticks)
        plt.legend(str(i))
        savefig("F:\\IJCAI-Koubei\\dataset\\cluster_everyweekday\\plot_1\\week" + str(weekday)+ "\\" + str(i) + "\\0_100_vali.jpg")

        df2 = df[(df[begin] >= 100) & (df[end] >= 100) & (df[begin] < 400) & (df[end] < 400)]
        df2 = df2.T
        df2.plot(kind='line', alpha=1, xticks=ticks)
        plt.legend(str(i))
        savefig("F:\\IJCAI-Koubei\\dataset\\cluster_everyweekday\\plot_1\\week" + str(weekday) + "\\" + str(i) + "\\100_400_vali.jpg")

        df3 = df[(df[begin] >= 400) | (df[end] >= 400)]
        df3=df3.T
        if df3.isnull==False:
            df3.plot(kind='line', alpha=1, xticks=ticks)
            plt.legend(str(i))
            savefig("F:\\IJCAI-Koubei\\dataset\\cluster_everyweekday\\plot_1\\week" + str(weekday) + "\\" + str(i) + "\\400_1200_vali.jpg")


def lastyeardata():
    data=read("cluster_everyweekday\\cluster_1\\weekday1")
    data=data[data['type']==0]
    lastyear=read("pivot_table3")
    lastyear=lastyear[['shop_id','20150907','20150914','20150921','20150928','20151012','20151019']]
    data=pd.merge(data,lastyear,on='shop_id')
    print data

def plot_list(week):
    if week==1:
        list=['shop_id','20160905', '20160912', '20160919', '20160926', '20161003', '20161010', '20161017', '20161024',
             '20161031']
    elif week==2:
        list=['shop_id','20160830', '20160906', '20160913', '20160920', '20160927', '20161004', '20161011', '20161018',
              '20161025']
    elif week==3:
        list=['shop_id','20160831', '20160907', '20160914', '20160921', '20160928', '20161005', '20161012', '20161019',
              '20161026']
    elif week==4:
        list=['shop_id', '20160901','20160908', '20160915', '20160922', '20160929', '20161006', '20161013', '20161020',
              '20161027']
    elif week==5:
        list=['shop_id', '20160902', '20160909', '20160916', '20160923', '20160930', '20161007', '20161014', '20161021',
             '20161028']
    elif week==6:
        list=['shop_id', '20160903', '20160910', '20160917', '20160924', '20161001', '20161008', '20161015', '20161022',
             '20161029']
    else:
        list=['shop_id', '20160904', '20160911', '20160918', '20160925', '20161002', '20161009', '20161016', '20161023',
             '20161030']
    return list

#有国庆的显示出来看
def justlook_has1001():
    week = 4
    type=0
    begin='20160901'
    data1 = read("cluster_test_everyweekday\\cluster_week"+str(week))
    data1 = data1[data1['type'] == type]
    data1 = data1.ix[:, 'shop_id':]
    data2 = read("pivot3_everyweekday\\pt_week_" + str(week))
    data2 = data2.ix[:,plot_list(week)]
    data = pd.merge(data1, data2, on='shop_id')


    df = data.ix[:, begin:]
    df1 = df[df <=100]
    df1 = df1.dropna()

    df1=df1.reset_index(drop=True)
    list=xrange(df1.shape[0])
    ranlist=random.sample(list,7)#随机采样

    df1=df1.ix[ranlist,:]

    print df1
    df1 = df1.T
    df1.plot(kind='line', alpha=1, xticks=[0,1,2,3,4,5,6,7,8])
    plt.legend(str('1'))
    plt.show()
justlook_has1001()




