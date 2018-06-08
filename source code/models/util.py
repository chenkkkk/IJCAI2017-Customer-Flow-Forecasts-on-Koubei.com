# coding=utf-8

import pandas as pd
from settings import *
from datetime import datetime

'''两个时间天数差'''
def deltaDays(t1, t2):
    d1 = datetime.strptime(t1, "%Y-%m-%d")
    d2 = datetime.strptime(t2, "%Y-%m-%d")
    return (d2 - d1).days

'''获取今天是星期几'''
def week_day(d):
    d1 = datetime.strptime(d, "%Y-%m-%d")
    return d1.isoweekday()-1  # 获取时间周几

'''是否是周中'''
def is_middle_day(week):
    if week==3:
        return 1
    return 0

'''判断当前星期是否是周末(周六,周日)'''
def is_weekend(week):
    if week>=5:
        return 1
    return 0

'''判断当前日期是一个月的第几天'''
def day_in_month(d):
    return int(d[-2:])

'''判断当前日期是一个月的第几周'''
def week_in_month(t1, shift_week):
    return int(datetime.strptime(t1, "%Y-%m-%d").strftime('%U')) - shift_week

'''判断当前日期是第几个月'''
def month_cnt(t1, shift_month):
    return int(t1[5:7]) - shift_month

'''判断当前日期是否是假期'''
def is_holiday(t):
    if t in holidays:
        return 1
    return 0

'''判断当前日期是否是补班日'''
def is_extra_work_day(t):
    if t in extra_work_days:
        return 1
    return 0

'''判断当前日期是否是暑假'''
def is_summer_vocation(t):
    if '2016-07-01' < t < '2016-09-01':
        return 1
    return 0

'''判断当前日期是否是非工作日'''
def is_no_work_day(d):
    d1 = datetime.strptime(d, "%Y-%m-%d")
    if d in holidays or ((d1.isoweekday() - 1)>=5 and d not in extra_work_days):
        return 1
    return 0

'''距离非工作日距离，升序'''
def no_work_day_dist_asc(days):
    rank = 0
    ranks = []
    pre_day = days[0]
    for day in days:
        if day==pre_day:
            rank += 1
        else:
            rank = 1
        ranks.append(rank)
        pre_day = day
    return ranks

'''距离非工作日距离，降序'''
def no_work_day_dist_desc(days):
    rank = 0
    ranks = []
    pre_day = days[-1]
    for day in days[::-1]:
        if day == pre_day:
            rank += 1
        else:
            rank = 1
        ranks.append(rank)
        pre_day = day
    return ranks[::-1]

'''时间戳与商店的笛卡尔积,防止零流量问题'''
def productShop(start="", end=""):
    print '######### inner product ing...'
    shop_info = pd.read_csv(base + 'shop_info.csv')
    df = pd.DataFrame(columns=['time_stamp', 'shop_id'])
    for shop_id in range(1, shop_info.shape[0] + 1):
        time_stamps = pd.DataFrame(pd.date_range(start=start, end=end).astype(str), columns=['time_stamp'])
        time_stamps['shop_id'] = shop_id
        df = df.append(pd.DataFrame(time_stamps, columns=['time_stamp', 'shop_id']), ignore_index=True)
    return df

if __name__ == '__main__':
    d1 = '2016-06-01'