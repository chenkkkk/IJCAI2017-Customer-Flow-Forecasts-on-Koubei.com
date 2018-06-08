# -*-coding:utf-8-*-
import pandas as pd
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation, metrics
from scipy.stats import pearsonr

#1.读数据
def read(strs):
    data=pd.read_csv("F:\\IJCAI-Koubei\\dataset\\"+strs+".csv",encoding="gbk")
    return data
#2.写数据
def write(data,str):
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv",index=None,encoding="gbk")
def good_bad(x):
    if x<=50:
        return 0
    elif (x>50) and (x<=100):
        return 1
    elif (x>100) and (x<=150):
        return 2
    elif (x>150) and (x<=200):
        return 3
    elif (x>200) and (x<=300):
        return 4
    elif x>300:
        return 5

#3.编码
def code_api():
    data=read("api_data").set_index('Date')
    data=data.stack().reset_index()
    data.rename(columns={'level_1':'city_name',0:'api'},inplace=True)
    print data
    print data[data.isnull().values==True]
    data.ix[:,'api']=data.ix[:,'api'].apply(good_bad)


    write(data,"code_api2")
# code_api()

def dates(begin,end,list):
    cols = pd.date_range(begin, end, freq=timedelta(days=1))
    if list != []:
        for date in list:
            cols=cols[cols!=date]
    cols = (cols.map(lambda x: int(x.strftime('%Y%m%d')))).tolist()
    return cols

def applyx(x):
    list=dates('20161101','20161114',[])
    list2=[2,3,4,5,6,7,1,2,3,4,5,6,7,1]
    for i in xrange(14):
        if x==list[i]:
            return list2[i]


def merge_data_api():
    for i in xrange(1,8,1):
        data=read("weather_regular\\week_"+str(i))
        data2=read("weather_regular\\weather_shop_data")
        print data.shape[0]
        data=pd.merge(data,data2,how='left',on=['shop_id','date'])
        print data.shape[0]

        data=data.set_index('shop_id')
        write(data,"weather_regular\\weekdata_"+str(i))

    # print data2

def pear():
    for week in xrange(1,8,1):
        data=read("weather_regular\\weekdata_"+str(week))
        data=data.dropna()
        listi=[]
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        for i in xrange(1,2001,1):
            xdata=data[data['shop_id']==i]
            listi.append(i)
            list1.append(pearsonr(xdata['label'],xdata['topTemp']))
            list2.append(pearsonr(xdata['label'], xdata['lowTemp']))
            list3.append(pearsonr(xdata['label'], xdata['weathercode']))
            list4.append(pearsonr(xdata['label'], xdata['code_wind']))
        df=pd.DataFrame({'shop_id':listi,
                         'ptopTemp':list1,
                         'plowTemp':list2,
                         'pweather':list3,
                         'pwind':list4
                         })
        print df
        write(df,"weather_regular\\wea_pearson_"+str(week))

def reg():
    a='label'
    b='weathercode'
    for week in xrange(3,8,1):
        data=read("weather_regular\\weekdata_"+str(week))
        data=data[(data['date']>20161008)|(data['date']<20160930)].dropna().set_index('shop_id')

        print data
        listi=[]
        listj=[]
        listk=[]

        for i in data.index.drop_duplicates():

            idata=data.ix[[i],:][[a,b]].T
            lent=idata.shape[1]
            if lent<4:
                continue
            for j in xrange(lent):
                idata[str(j)+'_cha']=idata.iloc[:,j+1]-idata.iloc[:,j]
            idata= idata.T
            idata=idata.ix['0_cha':,:]
            # print i
            # print idata

            xdata=idata[idata[b]>=2]
            ydata=idata[idata[b]<=-2]
            # print xdata
            # print ydata

            s=xdata.shape[0]
            l=xdata[xdata[a]>0].shape[0]
            k=xdata[xdata[a]<0].shape[0]

            s2=ydata.shape[0]
            l2=ydata[ydata[a]>0].shape[0]
            k2=ydata[ydata[a]<0].shape[0]

            if (l*l2>0)or (k*k2>0):
                continue
            if (l==0) or (s-l==0):
                listi.append(i)



        weadata=pd.DataFrame({'shopid_weather':listi})
        # print weadata
        write(weadata,"weather_regular\\WeatherStrongShop_"+str(week))

def addWeek():
    # data1=read("weather_regular\\WeatherStrongShop_"+str(1))
    # for week in xrange(1,8,1):
    #     data=read("weather_regular\\WeatherStrongShop_"+str(week))
    #     data1=pd.merge(data1,data,how='left',on='shopid_weather')
    #
    # data1 = data1[['shopid_weather']]
    # data1 = data1[(data1.shopid_weather < 101) | ((data1.shopid_weather > 199) & (data1.shopid_weather < 1001)) |
    #             ((data1.shopid_weather > 1899) & (data1.shopid_weather < 2001))]

    data1=read("sortweather_new_forb")
    data1 = data1[(data1['max(1,2)/5'] < 0.9) & (data1['(3)/5'] < 1)]
    print data1.columns

    for week in xrange(1, 8, 1):
        data = read("weather_regular\\weekdata_"+str(week)).set_index('shop_id')
        data = data[(data['date'] > 20161008) | (data['date'] < 20160930)]


        data= data[data.index.isin (data1.shop_id)]
        write(data,"weather_regular\\newshoudong_"+str(week))


def predict_join_weather_api(strs):
    data=read("Predict_Result\\"+strs)
    data=data.stack().reset_index()
    # print data
    data['level_0']=data['level_0'].map(lambda x:x+1)#索引加1为商店号
    data=data[data['level_1']!='shop_id']
    data=data.rename(columns={'level_0':'shop_id','level_1':'date',0:'label'})
    data['date']=data['date'].map(lambda x:int(x))
    # write(data,"weather_regular\\unstack")
    weather=read("weather_regular\\weather_shop_data2")
    api=read("api_regular\\shop_api_data")

    data=pd.merge(data,weather,on=['shop_id','date'])
    data=pd.merge(data,api,on=['shop_id','date'])


    data['week']=data['date'].apply(applyx)
    # data=data.set_index('shop_id')
    data=data[['label','shop_id','date','week','topTemp','lowTemp','weathercode','api']]

    return data
    # write(data,"weather_regular\\weather_predict_data2")



def concat_predict():
    pre=read("weather_regular\\weather_predict_data2").set_index('shop_id')
    print pre
    # print type(pre.iloc[10,2])
    for week in xrange(1,8,1):
        ipre=pre[pre['week']==week]

        data=read("weather_regular\\newshoudong_"+str(week)).set_index('shop_id')
        xdata=data.ix[:-1,:]
        # print xdata
        for i in data.index.drop_duplicates():
            idata=data.ix[[i],:]
            idata=pd.concat((idata,ipre.ix[[i],:]))
            xdata=pd.concat((xdata,idata))
        write(xdata,"weather_regular\\newshoudong_"+str(week))

import predict as pr
def pre():
    week=1
    data=read("weather_regular\\newshoudong_"+str(week))
    # print data[['shop_id']].drop_duplicates()
    data=data[(data['date']==20161107)|(data['date']==20161114)]
    pt=pd.pivot_table(data,index='shop_id',columns='date',values='label').reset_index()
    pt=pt.rename(columns={20161107:'a',20161114:'b'})
    # print pt.columns

    predict_data = read("Predict_Result\\0308\\shev_czl_7")


    predicts = pd.merge(predict_data, pt[['shop_id', 'a','b']], how='left', on='shop_id')
    print predicts
    pre1 = predicts[predicts['a'].isnull() == True]
    pre2 = predicts[predicts['a'].isnull() == False]
    print pre2
    pre2['20161107'] = pre2['a']
    pre2['20161114'] = pre2['b']

    data = pd.concat([pre1, pre2])

    print data
    pr.Write_Predict_Model(data, "0308\\shev_czl_7_weather")

def find_sortweather():
    pre = read("weather_regular\\weather_predict_data").set_index('shop_id')
    pre=pre[pre.weathercode<4]
    week=7
    data=read("weather_regular\\shoudong_"+str(week))
    pre=pre[pre.date.isin([20161113,20161106])]
    # pre=pre[pre.topTemp<12]
    # print pre[pre.index.isin(data.shop_id)].index
    # data=read("weather_regular\\shoudong_1")
    # print data['shop_id'].drop_duplicates()

    data1=read("weather_regular\\weather_shop_data2")#所有查看雨天中位数和晴天中位数
    data2=read("shop_date_label")
    data=pd.merge(data1,data2,on=['shop_id','date'])

    data['datetime']=pd.to_datetime(data.date.map(lambda x:str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8]))
    data['week']=data.datetime.map(lambda x:x.weekday()+1)
    print data
    for week in xrange(1,8,1):
        xdata=data[data['week']==week]
        xdata=xdata.groupby([xdata['shop_id'],xdata['weathercode']])['label'].quantile(0.5).reset_index()
        xdata=pd.pivot_table(xdata,index='shop_id',columns='weathercode',values='label').reset_index()
        xdata['(12/5)']=xdata[[1,2]].max(1)/xdata[5]
        xdata['(3/5)']=xdata[3]/xdata[5]
        data1=xdata.sort_values(['(12/5)'])
    # print data1
    # write(data,"sortweather_new")
        data1 = data1[(data1.shop_id < 101) | ((data1.shop_id > 199) & (data1.shop_id< 1001)) |
                      ((data1.shop_id > 1899) & (data1.shop_id < 2001))].set_index('shop_id')
    # print data1
        write(data1,"weather_regular\\sortweather_week_"+str(week))

def xishu(data,col,indexs,a,b,c,d,e,f,strs):

    data_1 = data[data.shop_id.isin(indexs)]
    data_2 = data[data.shop_id.isin(indexs) == False]

    if strs=='all':
        data_1=data.copy()
        data_2 = data.ix[:-1, :]

    code=[1,2,3,4,5,6]     #天气越大越好
    if str(col)=='api':
        code=[5,4,3,2,1,0] # api越小越好

    x1 = data_1[data_1[col] == code[0]]

    x2 = data_1[data_1[col] == code[1]]

    x3 = data_1[data_1[col] == code[2]]
    x4 = data_1[data_1[col] == code[3]]
    x5 = data_1[data_1[col] == code[4]]
    x6 = data_1[data_1[col] == code[5]]
    x1.ix[:, 'label'] = x1.ix[:, 'label'] * a
    x2.ix[:, 'label'] = x2.ix[:, 'label'] * b
    x3.ix[:, 'label'] = x3.ix[:, 'label'] * c
    x4.ix[:, 'label'] = x4.ix[:, 'label'] * d
    x5.ix[:, 'label'] = x5.ix[:, 'label'] * e
    x6.ix[:, 'label'] = x6.ix[:, 'label'] * f
    data_1 = pd.concat((x1, x2, x3, x4, x5, x6))

    data_1 = pd.concat((data_1, data_2))

    return data_1
def index_w(sort_weather,a,b):
    indexs_1 = sort_weather[(sort_weather['max(1,2)/5'] < b)
                            & (sort_weather['max(1,2)/5'] >= a)
                            & (sort_weather['(3)/5'] > 1)].shop_id.drop_duplicates()
    return indexs_1
def index_api(sort_api,a,b):
    indexs_2 = sort_api[(sort_api['(12/5)'] < b)
                            & (sort_api['(12/5)'] >= a)].shop_id.drop_duplicates()
    return indexs_2

def predict_weather_api_xishu():
    data = predict_join_weather_api("0312_ycl\\ycl_allwithout7_37_5_a_dengyu_b")

    data2 = data.copy()
    sort_weather=read("sortweather_new_forb")
    iall=sort_weather.shop_id.drop_duplicates()

    i_1 = index_w(sort_weather, 0.0, 0.7)
    i_2 = index_w(sort_weather, 0.7, 0.8)
    i_3 = index_w(sort_weather, 0.8, 0.9)
    i_4 = index_w(sort_weather, 0.9, 1.0)

    #越小表示越敏感，越敏感则系数幅度越大
    # ---weather所有乘系数
    data = xishu(data, 'weathercode', iall, 0.98, 0.99, 0.99
                 , 1.01, 1.01, 1.02,'all')
    # data=xishu(data,'weathercode',i_1,0.999,0.999,0.999,1.0,1.001,1.001)
    # data=xishu(data,'weathercode',i_2,0.998,0.998,0.999,1.0,1.001,1.001)
    # data=xishu(data,'weathercode',i_3,0.998,0.998,0.998,1.0,1.001,1.001)
    data['date']=data['date'].map(lambda x:str(x))
    data=data[['shop_id','date','label']]
    pt=pd.pivot_table(data,index='shop_id',columns='date',values='label').reset_index()
    pr.Write_Predict_Model(pt,"0312\\y_all_98_99_99_101_101_102")

    #---api
    # sort_api = read("sortapi_forb")
    # indexs = sort_api[sort_api['(12/5)'] < 0.6].shop_id.drop_duplicates()
    #
    # data = xishu(data, 'api', indexs, 0.96, 0.98, 0.99
    #              , 1.0, 1.0, 1.0,'all')
    # data['date'] = data['date'].map(lambda x: str(x))
    # data = data[['shop_id', 'date', 'label']]
    # pt = pd.pivot_table(data, index='shop_id', columns='date', values='label').reset_index()
    # # pr.Write_Predict_Model(pt, "0310\\api")






def temp():
    data1 = read("weather_xishu_vali_0859")
    test = read("test_weather_api_forb")
    data=data1.copy()
    high,low=HighLowtemp(data,'vali')

    for i in high.keys():
        data.ix[(data.date == i) &(data.shop_id.isin(high[i]))
        ,'label']*=1.01
    for j in low.keys():
        data.ix[(data.date == j) & (data.shop_id.isin(low[j]))
        , 'label'] *= 0.995

    print r.score(data1, test)
    print r.score(data, test)
	myweather=pd.read_csv("F:\\ijcai\\myweather_gbk.csv",encoding='gbk',names=['city_name','date','topTemp','lowTemp','weather','wind','windStrength'])
	myweather=myweather[(myweather['date']>'2016-10-31')&(myweather['date']<'2016-11-15')]
	myweather['cod']=myweather['weather'].map(mapping)
	myweather.drop(['topTemp','lowTemp','wind','windStrength','weather'],axis=1,inplace=True)
	myweather.set_index(['city_name','date'],inplace=True)
	myweather=myweather.unstack()
	myweather.columns = myweather.columns.droplevel()
	myweather=pd.merge(shop_info_want_area,myweather,how='left',left_on='city_name',right_index=True)
	myweather.set_index('shop_id',inplace=True)
	myweather.drop('city_name',inplace=True,axis=1)
	myweather.columns=range(1,15)

	daan=pd.read_csv("C:\\mdu.csv",header=None)
	daan.set_index(0,inplace=True)

	timess=pd.read_csv('C:\\mdu\\dataset\\shop_info_gbk.txt',
		names=['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name',
		'cate_2_name','cate_3_name'],encoding='gbk')
	timess.set_index('shop_id',inplace=True)

	is_kind=timess['cate_2_name']=='快餐'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.053
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.95

	is_kind=timess['cate_2_name']=='超市'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.05
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.95

	is_kind=timess['cate_2_name']=='便利店'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.02
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.96

	is_kind=timess['cate_2_name']=='休闲茶饮'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.057
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.946

	is_kind=timess['cate_2_name']=='小吃'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.03
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.97

	is_kind=timess['cate_2_name']=='休闲食品'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.03
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.95

	is_kind=timess['cate_2_name']=='烘焙糕点'

	daan[myweather.loc[is_kind,:]<1]=daan[myweather.loc[is_kind,:]<1]*1.02
	daan[myweather.loc[is_kind,:]>=1]=daan[myweather.loc[is_kind,:]>=0]*0.946




import Regulation as r

def vali_weather_api_xishu():
    sort_weather = read("sortweather_new_forb")
    # sort_api=read("sortapi_forb")
    # data=read("vali_weather_api_forb")
    data = read("vali\\vali")
    test=read("vali\\test")
    data1=data.copy()
    id=sort_weather.shop_id.drop_duplicates()

    # indexs = sort_weather.shop_id.drop_duplicates()
    # indexs=sort_api[sort_api['(12/5)']<0.3].shop_id.drop_duplicates()


    # i_1=index_w(sort_weather,0.0,0.7)
    # i_2 =index_w(sort_weather,0.7,0.8)
    # i_3 =index_w(sort_weather,0.8,0.9)
    # i_4 =index_w(sort_weather,0.9,1.0)


    # 0.08596996
    min =0.0856#0.0861

    list=[]
    for j in xrange(100, 101, 1):
        j = j * 0.01

        data = xishu(data1, 'weathercode', id, 0.91, 0.98, 0.95, 1.0,              1.03,1.04, 'all')
        s=r.score(data, test)
        list.append(s)
        if min > s:
            print j
            print s
            min = s
        # print   r.score(data, test)

    l=pd.DataFrame({'x':list})
    # write(l,"graph\\graphallweather_6")
    # data = xishu(data, 'weathercode', i_2, 1.05, 1.03, 0.94, 1.01, 1.08, 1.15, '')
    # data = xishu(data, 'weathercode', i_3, 1.0, 1.0, 1.04, 1.04, 1.03, 1.1, '')
    # data = xishu(data, 'weathercode', i_4, 0.92, 1.0, 0.99, 0.96, 1.02, 0.98, '')

    # list.append(r.score(data, test))
    # if r.score(data, test) < min:
    #     min = r.score(data, test)
    #     print i
    #     print r.score(data, test)

    write(data,"vali\\vali_08544")

  #api:
    # data = read("weather_xishu_vali_0859")
    # data = xishu(data, 'api', indexs, 0.0, 0.98, 1.0
    #              , 1.0, 1.0, 1.0,'')

    # print r.score(data1, test)
    # write(pd.DataFrame({'0':list}),"weather_regular\\canshu\\5_1")



def HighLowtemp(data,strs):

    lt = [20161114, 20161108, 20161109, 20161110, 20161111, 20161112, 20161113]
    high = {20161114: [], 20161108: [], 20161109: [], 20161110: [],
            20161111: [], 20161112: [], 20161113: []}
    low = high.copy()
    if strs=='vali':
        lt=[20161031,20161025,20161026,20161027,20161028,20161029,20161030]
        high={20161031:[],20161025:[],20161026:[],20161027:[],
               20161028: [], 20161029: [], 20161030: []}
        low=high.copy()

    for i in xrange(1,8,1):
        idata=data[data.week==i]
        idata=pd.pivot_table(idata,index=['shop_id'],
                          columns='date',
                          values='topTemp').reset_index()
        idata['losTemp']=idata.iloc[:,1]-idata.iloc[:,2]

        lowdata = idata[(idata.iloc[:, 1] < 25) &
                         (idata.iloc[:, 1] - idata.iloc[:, 2] > 10)]
        highdata = idata[(idata.iloc[:, 2] < 25) &
                         (idata.iloc[:, 1] - idata.iloc[:, 2] < (-10))]
        high[lt[i - 1]]=highdata['shop_id'].tolist()
        low[lt[i - 1]]=lowdata['shop_id'].tolist()

    return high,low