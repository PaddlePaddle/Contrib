#encoding=utf8
from datetime import datetime
import requests
import re
import datetime
import numpy as np
from numpy import mat
import pandas as pd
import time
import os
import csv


# 读取75个城市
city = pd.read_csv("./data/crawl_list.csv")
city_list = list(city.city)
num_list = list(city.num)
    
# 确诊时序数据定义
zxs = ['北京市','上海市','重庆市','天津市']
fan = ['县','区','省']
zxs_cut = ['provinceName','province_confirmedCount','updateTime']
city_cut = ['cityName','city_confirmedCount','updateTime']
structure = ['zone','count','time']


def index2data(df):
    begin = '2020-01-01 00:00:00'
    begin = time.mktime(time.strptime(begin,'%Y-%m-%d %H:%M:%S'))
    init = '1970-01-01 00:00:00'
    init = time.mktime(time.strptime(init,'%Y-%m-%d %H:%M:%S'))

    sub = begin - init
    df.index = pd.to_datetime(df.index+sub/(3600*24),unit='d')
    return df

   
    
def get_input():
    # data
    df = pd.read_csv('./data/mock_data', sep='\t', names=['date','迁出省份','迁出城市','迁入省份','迁入城市','人数'])
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df = df.set_index('date')
    df = df['2020']
    city_df = pd.read_csv('./data/crawl_list.csv')
    input_df = pd.DataFrame()

    # 筛选迁出城市
    out_df_wuhan = df[df['迁出城市'].str.contains('武汉')]
    for i in city_df['city']:
        # 筛选迁入城市
        in_df_i = out_df_wuhan[out_df_wuhan['迁入城市'].str.contains(i)]
        # 确保按时间升序
        # in_df_i.sort_values("date",inplace=True)
        in_df_i.reset_index(drop=True, inplace=True)
        input_df[i] = in_df_i['人数']
    input_df.columns = list(range(75))
    return input_df

        
def get_confirm():
    df = pd.read_csv("./data/DXYArea.csv") # 这里的DXYArea.csv要每日自动更新一下
    df_zxs = pd.DataFrame()
    df_city = df

    # 把四个直辖市挑出来单独处理
    for i in list(zxs):
        df_zxs = df_zxs.append(df[df['provinceName']==i])
        df_city=df_city[~ df_city['provinceName'].str.contains(i)]
    df_zxs = df_zxs[zxs_cut]
    df_city = df_city[city_cut]
    df_zxs.columns =  structure
    df_city.columns = structure
    df = df_zxs.append(df_city)

    # 去掉省、县、区
    for i in list(fan):
         df = df[~ df['zone'].str.contains(i)]
    #北京“市”
    df2 = pd.DataFrame((x.split('市')for x in df['zone']),index=df.index,columns=['city','none'])
    df = pd.merge(df,df2,right_index=True,left_index=True)
    df = df[['city','count','time']]
    df =df.sort_values(by='time',ascending=True)
    
    #处理确诊序列
    confirm = np.zeros((100,len(city_list)))
    confirm = pd.DataFrame(confirm,columns=num_list)

    for i in city_list:
        df_city = df[df['city']==i]
        df_city['time']= df_city['time'].str[:10]
        df_city['time'] = pd.to_datetime(df_city['time'], format ='%Y-%m-%d')
        df_city['time'] = (df_city['time'].values - np.datetime64('2020-01-24T00:00:00Z')) / np.timedelta64(1, 'h')

        step=24
        df_city['time'] = (df_city['time']/step).astype(int)
        df_city = df_city.drop_duplicates(['time'],keep='last')
        #print(df_city)
        for j in df_city['time']:
            confirm.iloc[int(j),city_list.index(i)] = df_city[df_city['time'] == j]['count'].iloc[0]


    confirm = (confirm.loc[~(confirm==0).all(axis=1), :])
    confirm.to_excel("test.xlsx")
    confirm=confirm.replace(0,np.nan)
    confirm=confirm.fillna(method='ffill',axis=0)
    confirm=confirm.replace(np.nan,0)
    
    #导入前24天数据
    insert = pd.read_csv("./data/insert.csv",index_col=0)
    insert = insert[:23]
    insert = pd.DataFrame(insert.values)
    confirm = pd.concat([insert,confirm]).reset_index(drop=True)

    return confirm[:-1]
    
def align_data(confirm, input):
    input = input[:len(confirm)-1]

    return confirm, input
    
    
def get_output(confirm):
    output=confirm.copy(deep=True)[22:]
    output = (output-output.shift(1))[1:]
    output[output < 0] = 0
    return output
    


def load_to_csv(input, output):
    input = index2data(input)
    output = index2data(output)
    
    input.to_csv("./data/input.csv")
    output.to_csv("./data/output.csv")
    
def main():
    # 用脚本更新每日确诊数据         
    input = get_input() # 这个是数据接口
    confirm = get_confirm() # 确诊序列规整

    confirm, input = align_data(confirm, input)# 对齐数据，使input比confirm少一天
    
    output = get_output(confirm)
    load_to_csv(input, output)
    
    
if __name__=='__main__':
    main()
    
