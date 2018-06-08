# -*-coding:utf-8-*-
import pandas as pd
from datetime import timedelta

def dates(begin,end,list,f):
    cols = pd.date_range(begin, end, freq=timedelta(days=f))
    if list != []:
        for date in list:
            cols=cols[cols!=date]
    cols = (cols.map(lambda x: x.strftime('%Y%m%d'))).tolist()
    return cols
#1.读数据
def read(strs):
    data=pd.read_csv("F:\\IJCAI-Koubei\\dataset\\"+strs+".csv",encoding="gbk")
    return data
#2.写数据
def write(data,str):
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv",encoding="gbk")
def col(week):
    if week==1:
        cols=dates('20160829','20161017',['20161003'],7)+['location_id','shop_id']
    elif week==2:
        cols=dates('20160830','20161011',['20161004'],7)+['location_id','shop_id']
    elif week==3:
        cols=dates('20160831','20161012',['20161005'],7)+['location_id','shop_id']
    elif week==4:
        cols=dates('20160901','20161013',['20161006'],7)+['location_id','shop_id']
    elif week==5:
        cols=dates('20160902','20161014',['20160930','20161007'],7)+['location_id','shop_id']
    elif week==6:
        cols=dates('20160903','20161015',['20161001','20161008'],7)+['location_id','shop_id']
    elif week==7:
        cols=dates('20160904','20161016',['20161002','20161009'],7)+['location_id','shop_id']
    return cols
def shop_Of_loc():
    
    for week in xrange(1, 8, 1):
        data = read("data_preprocessing\\train_week_" + str(week)).sort_values('shop_id')
        data = data[col(week)]
        countdata = data.groupby(data['location_id'])['shop_id'].count().reset_index()
        loc= countdata[countdata.shop_id>=2].location_id


        data=pd.pivot_table(data,index=['location_id','shop_id'])\
                            .stack(dropna=False)\
                            .reset_index()\
                            .rename(columns={'level_2':'date',0:'label'})
        # print data
        xdata = pd.DataFrame()
        for i in loc:
            idata=data[data.location_id==i]
            idata=pd.pivot_table(idata,index=['location_id','date'],\
                                 columns='shop_id',values='label').reset_index()
            cols=idata.columns.tolist()[2:]#保存shop_id
            rcol = []#保存占区域比
            # print idata
            # print cols
            # print cols[2:]
            for j in xrange(len(cols)):

                idata.ix[:,'r_'+str(cols[j])]=idata[cols[j]]\
                                              /idata.ix[:,cols].mean(1)
                # idata.ix['mean',:]=idata.ix[:,'r_'+str(cols[j])].mean(0)
                rcol.append('r_'+str(cols[j]))
            idata=idata[['date']+rcol].set_index('date').T
            
            cl=idata.columns.tolist()
            idata.ix[:,'cov']=idata.ix[:,cl].std(1)
            idata.ix[:,'mid']=idata.ix[:,cl].quantile(0.5,1,interpolation='nearest')
            idata.ix[:,'mean']=idata.ix[:,cl].mean(1)

            xdata=pd.concat((xdata,idata))

        xdata['mid'] = xdata['mid'].fillna(xdata['mean'])
        xdata=xdata.sort_values(['cov'])
        write(xdata,"Loc_Regular\\sortrate_"+str(week))
def data_week(data,i,j):
    # l = dates('20161018', '20161031', [], 1)
    if j==1:
        l1 = ['20161024'] + dates('20161018', '20161023', [], 1)
    elif j==2:
        l1 = ['20161031'] + dates('20161025', '20161030', [], 1)
    data=data[data.date==l1[i-1]]
    return data
def xishu(data,covs,weekday,j):
    sorts=read("Loc_Regular\\sortrate_"+str(weekday))
    sorts=sorts[sorts['cov']<covs]
    sorts.shop_id=sorts.shop_id.map(lambda x:float(x[2:]))
    sorts.ix[:,'week']=weekday

    data1=data[(data.week==weekday)&(data.shop_id.isin(sorts.shop_id))]
    data2=data[(data.week!=weekday)|(data.shop_id.isin(sorts.shop_id)==False)]


    col=data1.columns.tolist()

    # print data1
    # print xdata

    data1=pd.merge(data1,sorts[['shop_id','week','mid']],on=['shop_id','week'])
    # print data1
    data1.ix[:,'label']=data1.ix[:,'loclabel']*data1.ix[:,'mid']
    data1=data1[col]


    xdata = pd.concat((data1, data2))
    return xdata
import Regulation  as r

def vali_loc():
    loc=read("Merged_Data")[['shop_id','location_id']]
    loc=loc.drop_duplicates(['shop_id'])

    data=read("vali\\vali_08544")
    data.date = data.date.map(lambda x: str(x))
    test = read("vali\\test")
    test.date = test.date.map(lambda x: str(x))

    data=pd.merge(data,loc,how='left',on='shop_id')

    # print data
    print data

    locmean=data.groupby(['location_id','date'])['label'].\
        mean().reset_index().rename(columns={'label':'loclabel'})
    data1= data[['shop_id','location_id','date','label']]
    data1=pd.merge(data,locmean,how='left',on=['location_id','date'])
    print locmean
    min=0.0854
    # for i in xrange(1,2,1):
    #     i = i * 0.01
    #     data = xishu(data1, 0.06, 1, '')
    #     # data = xishu(data1, i, 2, '')
    #     # data = xishu(data1, i, 3, '')
    #     # data = xishu(data, 0.03, 4, '')
    #     # data = xishu(data, 0.04, 5, '')
    #     # data = xishu(data, 0.05, 6, '')
    #     # data = xishu(data, 0.03, 7, '')
    #     s=r.score(data, test)
    #     if s<min:
    #         print i
    #         print s
    #         min=s
    #     print s

    # for week in xrange(1,2,1):
    #     data=data_week(data1,week,1)
    #     print data


vali_loc()