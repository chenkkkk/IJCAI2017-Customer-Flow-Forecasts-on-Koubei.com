# -*-coding:utf-8-*-
import pandas as pd
from datetime import timedelta



#1.读数据
def read(strs):
    data=pd.read_csv("F:\\IJCAI-Koubei\\dataset\\"+strs+".csv",encoding="gbk")
    return data
#2.写数据
def write(data,str):
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv",encoding="gbk")

def api():
    data = read("code_api2")
    print data
def wash_api():

    data=read("Merged_Data")[['shop_id','city_name']]
    data=data.drop_duplicates(['shop_id']).reset_index(drop=True)
    # print data
    weather=read("code_api2")


    data=pd.merge(data,weather,how='right',on='city_name')
    data=data.sort_values('shop_id')
    data=data[['shop_id','Date','api']]
    # data['code_wind'] = data['code_wind'].fillna(1)
    data.rename(columns={'Date': 'date'}, inplace=True)
    data=data[data.shop_id.isin (xrange(1,2001,1))].sort_values(['shop_id','date'])
    # data=data[(data.date>20160101)&(data.date<20161201)]
    # print data[data.isnull().values==True]

    data=data.set_index('shop_id')
    print data
    # write(data, "shop_api_data")
wash_api()
import Weather_Air_regular as war
def predictaddweather():
    data=read("Predict_Result\\0308\\shev_czl_7")
    data=data.stack().reset_index()
    data['level_0']=data['level_0'].map(lambda x:x+1)
    data=data[data['level_1']!='shop_id']
    data=data.rename(columns={'level_0':'shop_id','level_1':'date',0:'label'})
    data['date']=data['date'].map(lambda x:int(x))
    # write(data,"api_regular\\unstack")
    weather=read("api_regular\\shop_api_data")

    data=pd.merge(data,weather,on=['shop_id','date'])

    data['week']=data['date'].apply(war.applyx)
    data=data.set_index('shop_id')
    data=data[['label','date','week','api']]

    print data
    # write(data,"api_regular\\api_predict_data2")


def find():


    data1=read("api_regular\\shop_api_data")#所有查看api中位数和api中位数
    data2=read("shop_date_label")
    data=pd.merge(data1,data2,on=['shop_id','date'])

    data['datetime']=pd.to_datetime(data.date.map(lambda x:str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8]))
    data['week']=data.datetime.map(lambda x:x.weekday()+1)
    print data
    for week in xrange(1,8,1):
        xdata=data[data['week']==week]
        xdata=xdata.groupby([xdata['shop_id'],xdata['api']])['label'].quantile(0.5).reset_index()
        xdata=pd.pivot_table(xdata,index='shop_id',columns='api',values='label').reset_index()
        xdata['(12/5)']=xdata[5]/xdata[[0,1,2]].quantile(0.5,axis=1)
        # xdata['(3/5)']=xdata[3]/xdata[5]
        data1=xdata.sort_values(['(12/5)'])
    # print data1
    # write(data,"sortapi_new")
        data1 = data1[(data1.shop_id < 101) | ((data1.shop_id > 199) & (data1.shop_id< 1001)) |
                      ((data1.shop_id > 1899) & (data1.shop_id < 2001))].set_index('shop_id')
    # print data1
        write(data1,"api_regular\\sortapi_week_"+str(week))
