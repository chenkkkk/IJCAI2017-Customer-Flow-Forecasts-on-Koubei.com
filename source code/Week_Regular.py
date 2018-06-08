# -*-coding:utf-8-*-
import pandas as pd
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation, metrics
from scipy.stats import pearsonr
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
    data.to_csv("F:\\IJCAI-Koubei\\dataset\\"+str+".csv",index=None,encoding="gbk")

def week():
    week1=read("data_preprocessing\\train_week_1")
    week1=week1.sort_values(['shop_id']).set_index('shop_id').ix[:,'20160905':]
    week2=read("data_preprocessing\\train_week_2")
    week2=week2.sort_values(['shop_id']).set_index('shop_id').ix[:,'20160906':]
    data=pd.concat((week1,week2),axis=1)
    list1=['20160905','20160912','20160919','20160926','20161010','20161017','20161024']
    list2=['20160906','20160913','20160920','20160927','20161011','20161018','20161025']
    for i in xrange(len(list1)):
        data[list2[i]+'/'+list1[i]]=data.ix[:,list2[i]]/data.ix[:,list1[i]]
    data['cov']=data.ix[:,list2[0]+'/'+list1[0]:list2[6]+'/'+list1[6]].std(axis=1)
    data['midrate']=data.ix[:,list2[4]+'/'+list1[4]:list2[6]+'/'+list1[6]].quantile(0.5, axis=1)
    data=data.sort_values(['cov'])

    write(data,"week_regular\\week2_week1")
import predict as pr
import Classifier as cf

def week2():
    list1 = ['20160905', '20160912', '20160919', '20160926', '20161010', '20161017', '20161024']
    list2 = ['20160906', '20160913', '20160920', '20160927', '20161011', '20161018', '20161025']
    data = read("week_regular\\week2_week1")
    data['cov2']=data.ix[:,list2[0]+'/'+list1[0]:list2[5]+'/'+list1[5]].std(axis=1)
    data=data.sort_values(['cov2'])
    data['midrate2']=data.ix[:,list2[3]+'/'+list1[3]:list2[5]+'/'+list1[5]].quantile(0.5, axis=1)
    data['20161025_2']=data.ix[:,'20161024']*data.ix[:,'midrate2']
    data=data[data['cov2']<0.06]
    data=data.dropna()
    data['score']=abs(data['20161025_2']-data['20161025'])/(data['20161025_2']+data['20161025'])
    data=data[data['score']<=0.05]
    return data

def add():
    list1 = ['20160905', '20160912', '20160919', '20160926', '20161010', '20161017', '20161024']
    list2 = ['20160906', '20160913', '20160920', '20160927', '20161011', '20161018', '20161025']
    data=read("week_regular\\week2_week1")
    data=data.sort_values(['shop_id']).set_index('shop_id')
    data = data[data['cov'] < 0.06]
    # print data
    data1=data.ix[:,[list2[4]+'/'+list1[4],list2[5]+'/'+list1[5],list2[6]+'/'+list1[6],'midrate']]
    data1=data1.T
    data1=data1.fillna(data1.mean()).T

    data=pd.concat((data.ix[:,:-1],data1.ix[:,'midrate']),axis=1)

    # write(data, "week_regular\\week2_week1_fillna")
    data['a']=data.ix[:,'20161031']*data.ix[:,'midrate']
    data=data.reset_index()
    shopid=week2().shop_id.tolist()
    data=data[data.shop_id.isin (shopid)]
    # print data

    predict_data=read("Predict_Result\\0302\\shev3").set_index('shop_id')
    pre=predict_data.copy()
    pre.ix[:,:]=0
    pre.ix[:,'20161101']=predict_data.ix[:,'20161101']
    pre=pre.reset_index()

    predicts = pd.merge(pre, data[['shop_id', 'a']], how='left', on='shop_id')
    # print predicts
    pre1 = predicts[predicts['a'].isnull() == True]
    pre2 = predicts[predicts['a'].isnull() == False]

    pre2['20161101'] = pre2['a']

    data = pd.concat([pre1, pre2])
    data.ix[:,'20161108']=data.ix[:,'20161101']
    print data
    pr.Write_Predict_Model(data, "0307\\lyd")
import Regulation as r
def col(week):
    if week==1:
        cols=dates('20160912','20161031',['20161003','20161010'],7)
    elif week==2:
        cols=dates('20160906','20161025',['20160927','20161004'],7)
    elif week==3:
        cols=dates('20160907','20161026',['20160928','20161005'],7)
    elif week==4:
        cols=dates('20160908','20161027',['20160929','20161006'],7)
    elif week==5:
        cols=dates('20160909','20161028',['20160930','20161007'],7)
    elif week==6:
        cols=dates('20160910','20161029',['20161001','20161008'],7)
    elif week==7:
        cols=dates('20160911','20161030',['20161002','20161009'],7)
    return cols
def weekdayOfthisweek():
    xdata=pd.DataFrame()
    for i in xrange(1,8,1):
        data=read("data_preprocessing\\train_week_"+str(i)).set_index('shop_id')
        data=data[col(i)]
        data=data.stack(dropna=False).reset_index()
        data.columns=['shop_id'+str(i),'weekday_'+str(i),'week'+str(i)]
        data=data.sort_values(['shop_id'+str(i),'weekday_'+str(i)]).reset_index(drop=True)
        xdata=pd.concat((xdata,data),axis=1)
    data=xdata.set_index('shop_id1')
    data=data[['week1','week2','week3','week4',
                 'week5', 'week6', 'week7']]#一共有七列，每个商家有6个样本，代表6周

    #求星期几占所有均值的比率
    for j in xrange(1,8,1):
        data.ix[:,'r_week_'+str(j)]=data.ix[:,'week'+str(j)]/data.ix[:,'week1':'week7'].mean(1)
    data = data[['r_week_1', 'r_week_2', 'r_week_3', 'r_week_4',
                 'r_week_5', 'r_week_6', 'r_week_7']]
    # print data
    columnlist=[0,1,2,3,4,5]*2001
    for k in xrange(1,8,1):
        xdata=data[['r_week_'+str(k)]]
        xdata.ix[:,'dijizhou']=columnlist
        xdata=xdata.reset_index()
        xdata=pd.pivot_table(xdata,index='shop_id1',
                          columns='dijizhou',
                          values='r_week_'+str(k))
        xdata.ix[:,'cov']=xdata.ix[:,:].std(axis=1)
        xdata.ix[:, 'mid'] = xdata.ix[:, 0:6].quantile(0.5,axis=1,interpolation='nearest')

        xdata.ix[:, 'mean'] = xdata.ix[:, 0:6].mean(axis=1)

        xdata=xdata.sort_values('cov').reset_index().rename(columns={'shop_id1':'shop_id'})

        write(xdata,"weekdayOfthisweek\\sortTest_"+str(k))
        data1=xdata.copy()
        data1 = data1[(data1.shop_id < 101) | ((data1.shop_id > 199) & (data1.shop_id < 1001)) |
                      ((data1.shop_id > 1899) & (data1.shop_id < 2001))]

        write(data1,"weekdayOfthisweek\\sortforbTest_"+str(k))
#找出哪周小于cov的数据的shop_id和比例中位数
def sorts(week,covs):
    sortx = read("weekdayOfthisweek\\sortTest_" +
                 str(week)).set_index('shop_id')
    sortx['mid'] = sortx['mid'].fillna(sortx['mean'])
    sortx=sortx[sortx['cov']<covs]
    return sortx.index,sortx[['mid']]
def xishu(data,i,r,strs):
    l=dates('20161101','20161114',[],1)
    l1=['20161107']+dates('20161101','20161106',[],1)
    l2=['20161114']+dates('20161108','20161113',[],1)
    # l=dates('20161018','20161031',[],1)
    # l1=['20161024']+dates('20161018','20161023',[],1)
    # l2=['20161031']+dates('20161025','20161030',[],1)
    indexs, sort = sorts(i, r)
    data2 = data[data.index.isin(indexs) == False]
    data = pd.concat((data[data.index.isin(indexs)], sort), axis=1)
    if strs==1:
        data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(axis=1)*data.ix[:,'mid']
    elif strs==2:
        data.ix[:, l2[i - 1]] = data.ix[:, l2].mean(axis=1) * data.ix[:, 'mid']
    data = pd.concat((data2, data[l]))
    return data
def Vali_week():
    data = read("weather_xishu_vali_0855")
    data.date=data.date.map(lambda x : str(x))
    test = read("test_weather_api_forb")
    test.date=test.date.map(lambda x:str(x))
    data1=data.copy()

    data1 = pd.pivot_table(data, index='shop_id', columns='date',
                        values='label')
    # data1=read("")
    min=0.0842124915876
    # min=0.09
    list=[]
    print data1
    for i in xrange(1,2,1):
        data = xishu(data1, 1, 0.2, 1)
        data = xishu(data, 2, 0.2, 1)  # 28,38,45,66
        data = xishu(data, 3, 0.2, 1)  # 28,46
        data = xishu(data, 4, 0.2, 1)  # 0.2
        data = xishu(data, 5, 0.15, 1)  # 0.15
        data = xishu(data, 6, 0.06, 1)
        xdata = xishu(data, 7, 0.06, 1)
    #   r.score(xdata, test)

        # data = xishu(data1, 1, 0.2, 1)
        # data = xishu(data, 1, 0.2, 2)  # 20,31,53,64
        # data = xishu(data, 2, 0.66, 1)  #28,38,45,66
        # data = xishu(data, 2, 0.75, 2)  # 28,66,75
        # data = xishu(data, 3, 0.28, 1)  #28,46
        # data = xishu(data, 3, 0.38, 2)  # 27,38
        # data = xishu(data, 4, 0.2, 1)   #0.2
        # data = xishu(data, 4, 0.22, 2)  # 22,42
        # data = xishu(data, 5, 0.15, 1)#0.15
        # data = xishu(data, 5, 0.18, 2)
        # data = xishu(data, 6, 0.07, 1)
        # data = xishu(data, 6, 0.07, 2)
        # data = xishu(data, 7, 0.07, 1)
        # data = xishu(data, 7, 0.07, 2)
    #
    #
    #     xdata = data.stack(dropna=False).reset_index().rename(columns={0: 'label'})
    #     list.append(r.score(xdata, test))
    #     print r.score(xdata, test)
    #     # if  r.score(xdata, test)<min:
    #     #     min=r.score(xdata, test)
    #     #     print i
    #     #     print min

    # ls=pd.DataFrame({'0':list})
    # print r.score(xdata, test)

    # write(ls,"graph\\graph_2_7")
    # for i in xrange(1,2,1):
    #    indexs,sort=sorts(i,0.2)#1,2,3
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(2,3,1):
    #    indexs,sort=sorts(i,0.28)#1,2,3
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(3,4,1):
    #    indexs,sort=sorts(i,0.28)#1,2,3
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(4, 5, 1):
    #     indexs, sort = sorts(i, 0.2)  # 1,2,3
    #     data2 = data[data.index.isin(indexs) == False]
    #     data = pd.concat((data[data.index.isin(indexs)], sort), axis=1)
    #
    #     data.ix[:, l1[i - 1]] = data.ix[:, l1].mean(1) * data.ix[:, 'mid']
    #     # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #     data = pd.concat((data2, data[l]))
    # for i in xrange(5, 6, 1):
    #     indexs, sort = sorts(i, 0.15)  # 1,2,3
    #     data2 = data[data.index.isin(indexs) == False]
    #     data = pd.concat((data[data.index.isin(indexs)], sort), axis=1)
    #
    #     data.ix[:, l1[i - 1]] = data.ix[:, l1].mean(1) * data.ix[:, 'mid']
    #     # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #     data = pd.concat((data2, data[l]))
    # for i in xrange(6, 7, 1):
    #
    #     indexs, sort = sorts(i, 0.05)  # 1,2,3
    #     data2 = data[data.index.isin(indexs) == False]
    #     data = pd.concat((data[data.index.isin(indexs)], sort), axis=1)
    #
    #     data.ix[:, l1[i - 1]] = data.ix[:, l1].mean(1) * data.ix[:, 'mid']
    #     # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #     data = pd.concat((data2, data[l]))
    # for i in xrange(7, 8, 1):
    #
    #     indexs, sort = sorts(i, 0.08)  # 1,2,3
    #     data2 = data[data.index.isin(indexs) == False]
    #     data = pd.concat((data[data.index.isin(indexs)], sort), axis=1)
    #
    #     data.ix[:, l1[i - 1]] = data.ix[:, l1].mean(1) * data.ix[:, 'mid']
    #     # data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #     data = pd.concat((data2, data[l]))
    #
    # #-----------------第二周
    # for i in xrange(1,4,1):
    #    indexs,sort=sorts(i,0.2)#1,2,3
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    # data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(4,5,1):
    #    indexs,sort=sorts(i,0.15)#4
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    # data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(5,6,1):
    #    indexs,sort=sorts(i,0.2)#5
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    # data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(6,7,1):
    #    indexs,sort=sorts(i,0.05)#6
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    # data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))
    # for i in xrange(7,8,1):
    #    indexs,sort=sorts(i,0.08)#7
    #    data2=data[data.index.isin(indexs)==False]
    #    data=pd.concat((data[data.index.isin(indexs)],sort),axis=1)
    #
    #    # data.ix[:, l1[i - 1]]=data.ix[:,l1].mean(1)*data.ix[:,'mid']
    #    data.ix[:, l2[i - 1]]=data.ix[:,l2].mean(1)*data.ix[:,'mid']
    #    data=pd.concat((data2,data[l]))


def Predict_week():
    # data1=read("Predict_Result\\0312\\mingan_baseOnalltianqixishu")
    data1=read("Predict_Result\\0310-ycl\\ycl_weather_all_1")
    data1=data1.set_index('shop_id')
    data = xishu(data1, 1, 0.15, 1)
    data = xishu(data, 2, 0.15, 1)  # 28,38,45,66
    data = xishu(data, 3, 0.15, 1)  # 28,46
    data = xishu(data, 4, 0.15, 1)  # 0.2
    data = xishu(data, 5, 0.12, 1)  # 0.15
    data = xishu(data, 6, 0.06, 1)
    data = xishu(data, 7, 0.06, 1)


    data=data.reset_index()
    data=data.sort_values(['shop_id'])
    print data
    pr.Write_Predict_Model(data,"0313\\weekReugular1baseOn817")

def pre0314():
    data=read("Predict_Result\\0313\\weekReugular1baseOn817")
    data.ix[:, '20161111']=data.ix[:,'20161111']*1.1
    pr.Write_Predict_Model(data, "0313\\w1111yd")






