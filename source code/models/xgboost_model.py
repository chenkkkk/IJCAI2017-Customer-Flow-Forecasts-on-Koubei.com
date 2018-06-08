# coding=utf-8

import pandas as pd
from util import *
import xgboost as xgb
import time

'''特征工程One-Hot'''
def featureEngineering(data):
    original_cols = data[['shop_id', 'time_stamp', 'week_day']]
    data_trans = pd.get_dummies(data, columns=['week_day'])
    data_trans = pd.merge(data_trans, original_cols, on=['shop_id', 'time_stamp'], how='left')
    return data_trans

'''日期特征'''
def date_feature(df):
    shift_week = int(datetime.strptime(time_range['start_train'], "%Y-%m-%d").strftime('%U')) - 1
    shift_month = int(time_range['start_train'][5:7]) - 1

    df['week_day'] = df['time_stamp'].apply(lambda t:week_day(t))                       # 星期
    df['week_day_inv'] = df['time_stamp'].apply(lambda t: 7-week_day(t))                # 星期逆序
    df['is_weekend'] = df['week_day'].apply(lambda t: is_weekend(t))                    # 周末
    df['is_middle_day'] = df['week_day'].apply(lambda t: is_middle_day(t))              # 周四
    df['is_extra_work_day'] = df['time_stamp'].apply(lambda t: is_extra_work_day(t))    # 补班
    df['is_holiday'] = df['time_stamp'].apply(lambda t: is_holiday(t))                  # 长假
    df['is_summer_vocation'] = df['time_stamp'].apply(lambda t: is_summer_vocation(t))  # 暑假
    df['is_no_work_day'] = df['time_stamp'].apply(lambda t: is_no_work_day(t))          # 非工作日
    df['no_work_day_dist_asc'] = no_work_day_dist_asc(df['is_no_work_day'].values)      # 距非工作日距离升序
    df['no_work_day_dist_desc'] = no_work_day_dist_desc(df['is_no_work_day'].values)    # 距非工作日距离逆序
    df['delta_days'] = df['time_stamp'].apply(lambda t: 1+deltaDays(time_range['start_train'], t)) # 第几天
    df['week_in_month'] = df['time_stamp'].apply(lambda t: week_in_month(t, shift_week))# 第几周
    df['month_cnt'] = df['time_stamp'].apply(lambda t: month_cnt(t, shift_month))       # 第几月

    return df

def feature_extract():
    print '################## extract feature '
    # 1. 店铺信息
    print '######### shop_info'
    shop_info = pd.read_csv(base + "shop_info.csv")
    feature = shop_info[['shop_id', 'city_name']]

    # 2. 移除国庆及前后一天
    print '######### remove 10.1'
    (start, end) = (time_range['start_train'], time_range['end_test'])
    time_stamps = productShop(start, end)
    feature = pd.merge(time_stamps, feature, on=['shop_id'], how='left')
    feature = feature[~feature['time_stamp'].isin(holiday_10_1)]

    # 3. 日期特征
    print '######### date feature'
    feature = date_feature(feature)

    # 4. 天气特征
    print '######### weather feature'
    weather_f = pd.read_csv(base + 'ijcai_weather.csv')
    weather_f = weather_f.drop(['topTemp', 'lowTemp'], axis=1)
    feature = pd.merge(feature, weather_f, on=['city_name', 'time_stamp'], how='left').fillna(0)

    # 5. 特征工程One-Hot
    feature = featureEngineering(feature)
    feature.to_csv(base + 'feature\\feature_all.csv', index=False)

def splitData():
    data = pd.read_csv(base + 'feature\\feature_all.csv')
    flow_cnt = pd.read_csv(base + 'dataset\\flow_cnt.csv')

    train = data[(data['time_stamp'] >= time_range['start_train']) & (data['time_stamp'] <= time_range['end_train'])]
    test  = data[(data['time_stamp'] >= time_range['start_test'] ) & (data['time_stamp'] <= time_range['end_test'])]
    train = pd.merge(train, flow_cnt, on=['shop_id', 'time_stamp'], how='left').fillna(0)

    train.to_csv(base + 'feature\\feature_train.csv', index=False)
    test.to_csv(base + 'feature\\feature_test.csv', index=False)


'''模型训练'''
def trainData(train_path, paras, num_round):
    dtrain = pd.read_csv(train_path)
    grouped = dtrain.groupby(['shop_id'])
    regs = {}
    for shop_id,group in grouped:
        print '#################################### train shop ' + str(shop_id)
        train_y = group['flow_cnt']
        train_x = group.drop(['shop_id', 'city_name', 'time_stamp', 'flow_cnt'], axis=1)
        train_matrix = xgb.DMatrix(train_x.values, label=train_y.values, feature_names=train_x.columns)
        watchlist = [(train_matrix, 'train')]
        model = xgb.train(paras, train_matrix, num_boost_round=num_round, evals=watchlist)
        regs[shop_id] = model
    return regs

'''模型预测'''
def predictData(test_path, regs):
    dtest = pd.read_csv(test_path)
    grouped = dtest.groupby(['shop_id'])
    preds_lst = []
    for shop_id, group in grouped:
        print '######### predict shop ' + str(shop_id) + ' ing...'
        test_x = group.drop(['shop_id', 'city_name', 'time_stamp'], axis=1)
        test_matrix = xgb.DMatrix(test_x.values, feature_names=test_x.columns)
        pred_lst = [int(shop_id)]
        preds = regs[shop_id].predict(test_matrix)#, ntree_limit=regs[shop_id].best_ntree_limit
        for pred in preds:
            if int(pred) < 0:
                pred_lst.append(10)
            else:
                pred_lst.append(int(pred))
        preds_lst.append(pred_lst + pred_lst[1:]) #复制一周
    pd.DataFrame(preds_lst, columns=None).to_csv(base + 'prediction.csv', index=False, header=None)

if __name__ == '__main__':
    time1 = time.time()

    # feature_extract()
    # splitData()

    train_path = base + 'feature\\' + 'feature_train.csv'
    test_path  = base + 'feature\\' + 'feature_test.csv'

    init_param = {
        'nthread': 8,
        'silent': 1,
        'objective': 'reg:linear',
        'seed': 10,
        'max_depth': 5,#5
        # 'gamma':0.2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.05,
        'min_child_weight': 1#16
    }

    regs = trainData(train_path, init_param, num_round=500)
    predictData(test_path, regs)

    time2 = time.time()
    print "running time: " + str((time2 - time1) * 1.0 / 60)

