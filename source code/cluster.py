#-*-coding:utf-8-*-
import pandas as pd
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
#1.读数据
def read(str):
    data=pd.read_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv", encoding="gbk")
    return data
#2.写数据
def write(data,str):
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv", encoding="gbk")
#设置一个类似于全局的数据，免得每次都读，这样我只读一次
#注：Train_Data是date小于2016-10-18的数据，Vali_Data是date大于等于2016-10-18的数据
Train_Data = read("train\\Train_Data")

Vali_Data = read("vali\\Vali_Data")
mod_vali=Vali_Data[['shop_id','date','week']]#提交的模板
pivot3=read("pivot_table3")#验证集
pivot3_vali=pivot3[['shop_id','20161018','20161019','20161020','20161021','20161022','20161023','20161024',
              '20161025','20161026','20161027','20161028','20161029','20161030','20161031']]
valiy1y2=read("vali\\vali_y1y2")[['shop_id','week','y1','y2']]
def code_weekday(weekday):
    data=read("pivot3_everyweekday\\pt_week_"+str(weekday))   #读取周几的数据
    if weekday==1:
        cols = pd.date_range('20160905', '20161017', freq=timedelta(days=7))
    elif weekday == 2:
        cols = pd.date_range('20160830', '20161011', freq=timedelta(days=7))
    elif weekday == 3:
        cols = pd.date_range('20160831', '20161012', freq=timedelta(days=7))
    elif weekday == 4:
        cols = pd.date_range('20160901', '20161013', freq=timedelta(days=7))
    elif weekday == 5:
        cols = pd.date_range('20160902', '20161014', freq=timedelta(days=7))
    elif weekday == 6:
        cols = pd.date_range('20160903', '20161015', freq=timedelta(days=7))
    else:
        cols = pd.date_range('20160904', '20161016', freq=timedelta(days=7))
    cols=cols[(cols>pd.to_datetime('20161008')) | (cols<pd.to_datetime('20160930'))]#排除掉国庆节异常数据
    cols=(cols.map(lambda x:x.strftime('%Y%m%d'))).tolist()
    cols.append('shop_id')
    data=data[cols]

    num=data.shape[1]-2                             #编码的个数
    for i in xrange(num):
        data[str(i+1)]= data.iloc[:,i+1]-data.iloc[:,i]
        data[str(i+1)]=data[str(i+1)].apply(CODE_012)     #编码

    Kmeans(data,weekday)


# code_weekday(1)
# code_weekday(2)
# code_weekday(3)
# code_weekday(4)
# code_weekday(5)
# code_weekday(6)
# code_weekday(7)
#3.函数,编码
def CODE_01(x):
    if x<0:
        return 0
    else:
        return 1
#4.将8月30日至10月17日每周一（或二至七）进行编码（当天大于上周这天为0，其它为1）
def code_weekday_2(weekday):
    # data=read("pivot3_everyweekday\\pt_week_"+str(weekday))   #读取周几的数据
    data=read("data_preprocessing\\fillna_week_"+str(weekday))
    if weekday==1:
        cols = pd.date_range('20160912', '20161031', freq=timedelta(days=7))
    elif weekday == 2:
        cols = pd.date_range('20160906', '20161025', freq=timedelta(days=7))
    elif weekday == 3:
        cols = pd.date_range('20160907', '20161026', freq=timedelta(days=7))
    elif weekday == 4:
        cols = pd.date_range('20160908', '20161027', freq=timedelta(days=7))
    elif weekday == 5:
        cols = pd.date_range('20160902', '20161028', freq=timedelta(days=7))
    elif weekday == 6:
        cols = pd.date_range('20160903', '20161029', freq=timedelta(days=7))
    else:
        cols = pd.date_range('20160904', '20161030', freq=timedelta(days=7))
    cols=cols[(cols>pd.to_datetime('20161009')) | (cols<pd.to_datetime('20160930'))]#排除掉国庆节异常数据
    cols=(cols.map(lambda x:x.strftime('%Y%m%d'))).tolist()
    cols.append('shop_id')

    data=data[cols]
    data['week']=weekday
    num=data.shape[1]-3                             #编码的个数
    for i in xrange(num):
        data[str(i+1)]= data.iloc[:,i+1]-data.iloc[:,i]
        data[str(i+1)]=data[str(i+1)].apply(CODE_01)     #编码

    return data
#5.转换为前一天和后一天
def date_To_y1y2(x):
    if x>='2016-10-18' and x<='2016-10-24':
        return 'y1'
    else:
        return 'y2'
#6.聚类all
def cluster_week_all_8():
    data1=code_weekday_2(1)
    data2=code_weekday_2(2)
    data3=code_weekday_2(3)
    data4=code_weekday_2(4)
    data5=code_weekday_2(5)
    data6=code_weekday_2(6)
    data7=code_weekday_2(7)
    data2.columns=data1.columns
    data3.columns = data1.columns
    data4.columns = data1.columns
    data5.columns = data1.columns
    data6.columns = data1.columns
    data7.columns = data1.columns
    list=[data1,data2,data3,data4,data5,data6,data7]
    data=pd.concat(list)


    feature = data.ix[:, '1':].drop_duplicates()
    joincol = feature.columns.tolist()
    feature = feature.reset_index(drop=True)
    feature['type']=feature.index

    data=pd.merge(data,feature,on=joincol)
    print data
    write(data,"cluster_2\\cluster_type8")

#7.聚类everyweek
def cluster_every_week_8(week):
    data=code_weekday_2(week)
    feature = data.ix[:, '4':'6'].drop_duplicates()
    joincol = feature.columns.tolist()
    feature = feature.reset_index(drop=True)
    feature['type']=feature.index

    data=pd.merge(data,feature,on=joincol)
    print data
    write(data,"cluster_test_everyweekday\\cluster_week"+str(week))


def collist(weekday):

    if weekday == 1:
        cols = pd.date_range('20160620', '20161031', freq=timedelta(days=7))
    elif weekday == 2:
        cols = pd.date_range('20160614', '20161025', freq=timedelta(days=7))
    elif weekday == 3:
        cols = pd.date_range('20160615', '20161026', freq=timedelta(days=7))
    elif weekday == 4:
        cols = pd.date_range('20160616', '20161027', freq=timedelta(days=7))
    elif weekday == 5:
        cols = pd.date_range('20160610', '20161028', freq=timedelta(days=7))
    elif weekday == 6:
        cols = pd.date_range('20160611', '20161029', freq=timedelta(days=7))
    else:
        cols = pd.date_range('20160612', '20161030', freq=timedelta(days=7))
    cols = cols[(cols > pd.to_datetime('20161009')) | (cols < pd.to_datetime('20160930'))]  # 排除掉国庆节异常数据
    cols = (cols.map(lambda x: x.strftime('%Y%m%d'))).tolist()
    return cols

#7.画图保存8个类
def plot_cluster(weekday):
    cols=collist(weekday)
    cols.append('shop_id')
    ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    week=read("cluster_test_everyweekday\\cluster_week"+str(weekday))
    week=week[['shop_id','type']]
    from_6=read("data_preprocessing\\fillna_week_"+str(weekday))
    week=pd.merge(week,from_6,on='shop_id')

    cols = collist(weekday)
    for i in xrange(8):
        weeki=week[week['type']==i]
        weeki=weeki.reset_index(drop=True)
        print weeki
        df = weeki.ix[:,cols]
        plt.legend('x')
        # 筛选出100以内的,100至400的，大于400，好观察
        df1=df[(df.ix[:,0]<100) & (df.ix[:,1]<100)&(df.ix[:,2]<100)&(df.ix[:,3]<100)
        &(df.ix[:,4]<100)&(df.ix[:,5]<100)&(df.ix[:,6]<100)&(df.ix[:,7]<100) &
               (df.ix[:,8]<100)&(df.ix[:,9]<100)&(df.ix[:,10]<100)
        &(df.ix[:,11]<100)&(df.ix[:,12]<100)&(df.ix[:,13]<100)&(df.ix[:,14]<100) &
               (df.ix[:,15]<100)&(df.ix[:,16]<100)&(df.ix[:,17]<100)
        &(df.ix[:,18]<100)]
        df1=df1.reset_index(drop=True)
        df1=df1.iloc[2:10,:]
        df1 = df1.T
        df1.plot(kind='line', alpha=1, xticks=ticks,yticks=[0,10,20,30,40,50,60,70,80,90,100])
        plt.legend(str(i))
        savefig("F:\\IJCAI-Koubei\\dataset\\cluster_test_everyweekday\\plot\\week_"+str(weekday)+ "\\type_"+
                str(i) + "\\from6_0_100_k.jpg")

        df2= df[(df.ix[:, 0] < 400) & (df.ix[:, 1] < 400) & (df.ix[:, 2] < 400) & (df.ix[:, 3] < 400)
                 & (df.ix[:, 4] < 400) & (df.ix[:, 5] < 400) & (df.ix[:, 6] < 400) & (df.ix[:, 7] < 400) &
                 (df.ix[:, 8] < 400) & (df.ix[:, 9] < 400) & (df.ix[:, 10] < 400)
                 & (df.ix[:, 11] < 400) & (df.ix[:, 12] < 400) & (df.ix[:, 13] < 400) & (df.ix[:, 14] < 400) &
                 (df.ix[:, 15] < 400) & (df.ix[:, 16] < 400) & (df.ix[:, 17] < 400)
                 & (df.ix[:, 18] < 400)
                 & (df.ix[:, 0] >= 100) & (df.ix[:, 1] >= 100) & (df.ix[:, 2] >= 100) & (df.ix[:, 3] >= 100)
                 & (df.ix[:, 4] >= 100) & (df.ix[:, 5] >= 100) & (df.ix[:, 6] >= 100) & (df.ix[:, 7] >= 100) &
                 (df.ix[:, 8] >= 100) & (df.ix[:, 9] >= 100) & (df.ix[:, 10] >= 100)
                 & (df.ix[:, 11] >= 100) & (df.ix[:, 12] >= 100) & (df.ix[:, 13] >= 100) & (df.ix[:, 14] >= 100) &
                 (df.ix[:, 15] >= 100) & (df.ix[:, 16] >= 100) & (df.ix[:, 17] >= 100)
                 & (df.ix[:, 18] >= 100)]
        df2 = df2.reset_index(drop=True)
        df2 = df2.iloc[2:10, :]
        df2 = df2.T
        df2.plot(kind='line', alpha=1, xticks=ticks,yticks=[100,150,200,250,300,350,400])
        plt.legend(str(i))
        savefig("F:\\IJCAI-Koubei\\dataset\\cluster_test_everyweekday\\plot\\week_"+str(weekday)+ "\\type_"+
                str(i) + "\\from6_100_400_k.jpg")

        df3 = df[(df.ix[:, 0] >= 400) & (df.ix[:, 1] >= 400) & (df.ix[:, 2] >= 400) & (df.ix[:, 3] >= 400)
                 & (df.ix[:, 4] >= 400) & (df.ix[:, 5] >= 400) & (df.ix[:, 6] >= 400) & (df.ix[:, 7] >= 400) &
                 (df.ix[:, 8] >= 400) & (df.ix[:, 9] >= 400) & (df.ix[:, 10] >= 400)
                 & (df.ix[:, 11] >= 400) & (df.ix[:, 12] >= 400) & (df.ix[:, 13] >= 400) & (df.ix[:, 14] >= 400) &
                 (df.ix[:, 15] >= 400) & (df.ix[:, 16] >= 400) & (df.ix[:, 17] >= 400)
                 & (df.ix[:, 18] >= 400)]
        df3 = df3.reset_index(drop=True)
        df3 = df3.iloc[2:10, :]
        df3=df3.T
        if df3.isnull==False:
            df3.plot(kind='line', alpha=1, xticks=ticks)
            plt.legend(str(i))
            savefig("F:\\IJCAI-Koubei\\dataset\\cluster_test_everyweekday\\plot\\week_"+str(weekday)+ "\\type_"+
                    str(i) + "\\from6_400_1200_k.jpg")
for i in xrange(1,8,1):
    plot_cluster(i)
#8.验证得分
def score_with_week(data,list_ab,list_date):
    eva=pd.merge(data,valiy1y2,on=['shop_id','week'])
    eva = eva.dropna()
    for i in xrange(len(list_ab)):
        if i ==0:
            s1 = sum(abs((eva[list_ab[i]] - eva[list_date[i]]) / (eva[list_ab[i]] + eva[list_date[i]]) ))
        else:
            s1 += (sum(abs((eva[list_ab[i]] - eva[list_date[i]]) / (eva[list_ab[i]] + eva[list_date[i]]))))
    s1 = s1 / ((eva.shape[0])*len(list_ab))
    print s1
#每个类别调参验证
def regular():
    data=read("cluster_2\\cluster_type8")

    t0=data[data['type']==0]

    t0=t0[t0['week']==4]
    # t0=t0.ix[:,'20160919':'20161017']
    # t0 = t0[(t0['20160919'] < 100) & (t0['20160926'] < 100) & (t0['20161010'] < 100) & (t0['20161017']< 100)]
    # t0=t0.T
    # t0.plot(kind='line', alpha=1, xticks=[0,1,2,3])
    # plt.legend(str(1))

    c1=0.05
    c2=0.2
    c3=0.35
    c4=0.4
    t0.ix[:,'a']=t0.ix[:,'20160919']*c1+t0.ix[:,'20160926']*c2+t0.ix[:,'20161010']*c3+t0.ix[:,'20161017']*c4
    score_with_week(t0,['a'],['y1'])

    # data = data[data['type'] == 3]

    t0 = data[data['week'] == 3]

    t0.ix[:,'a']=t0.ix[:,'20160919']*c1+t0.ix[:,'20160926']*c2+t0.ix[:,'20161010']*c3+t0.ix[:,'20161017'] * c4

    score_with_week(t0, ['a'], ['y1'])

