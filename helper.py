# -*- coding: UTF-8 -*-
import os
from datetime import datetime, timedelta
from fa.helper import single_news_to_representation, get_stock_date, get_date_return,formatDateString
from ta.helper import time_series_normalization
from keras.preprocessing import sequence
import tushare as ts
import numpy as np
import pandas as pd
import re

__author__ = 'shenxiangxiang@gmail.com'


def get_previous_stock_data(dt=None, td=400):
    """
    :param dt: date time of the news
    :param td: time delta, number days to backtrack, default 400
    :return:
    """
    if dt is None:
        dt = datetime.today()
    dt2 = dt - timedelta(days=td)
    # logger.info(self.formatDateString(dt2))
    stock_data = ts.get_hist_data('sh', start=formatDateString(dt2),
                                  end=formatDateString(dt), retry_count=10)

    stock_data = stock_data.as_matrix(['open', 'high', 'low', 'p_change', 'volume', 'close'])
    stock_data = stock_data[stock_data.shape[0]::-1, :]
    stock_data, _ = time_series_normalization(stock_data, mode=1)
    return stock_data


def read_news_and_generate_training_set(file_name, max_len_news=200, max_len_time_series=400, year='2015年'):
    one_day_news = []
    base_time = datetime(1991, 12, 20, 0, 0)
    news_headline = ""
    X1 = None
    X2 = None
    y = None
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) < 1:  # skip empty line
                continue
            if re.match(r'^\d+月\d+日 \d+:\d+$', line):
                line_time = datetime.strptime(year + line, '%Y年%m月%d日 %H:%M')
                line_time = get_stock_date(line_time)

                if (line_time - base_time).days >= 1 and len(one_day_news) == 0:
                    base_time = line_time
                    one_day_news.append(single_news_to_representation(news_headline))
                elif (line_time - base_time).days >= 1 and len(one_day_news) > 0:
                    # first convert news to x1
                    tmp = np.vstack(one_day_news)
                    tmp = tmp[np.newaxis, :, :]
                    tmp = sequence.pad_sequences(tmp, maxlen=max_len_news, dtype='float32')
                    X1 = tmp if X1 is None else np.vstack((X1, tmp))
                    # second, get stock data x2
                    tmp1 = get_previous_stock_data(base_time-timedelta(days=1))
                    tmp1 = tmp1[np.newaxis, :, :]
                    tmp1 = sequence.pad_sequences(tmp1, maxlen=max_len_time_series, dtype='float32')
                    X2 = tmp1 if X2 is None else np.vstack((X2, tmp1))
                    # third, return
                    p_change = get_date_return(base_time, max_day_try=15)
                    tmp = [p_change]
                    y = tmp if y is None else np.vstack((y, tmp))
                    base_time = line_time
                    one_day_news = list()
                    one_day_news.append(single_news_to_representation(news_headline))
                else:
                    one_day_news.append(single_news_to_representation(news_headline))
            else:
                news_headline = line
    # end with open file
    if X1 is not None:
        tmp = np.vstack(one_day_news)
        tmp = tmp[np.newaxis, :, :]
        tmp = sequence.pad_sequences(tmp, maxlen=max_len_news, dtype='float32')
        X1 = np.vstack((X1, tmp))
        tmp1 = get_previous_stock_data(base_time - timedelta(days=1))
        tmp1 = tmp1[np.newaxis, :, :]
        tmp1 = sequence.pad_sequences(tmp1, maxlen=max_len_time_series, dtype='float32')
        X2 = np.vstack((X2, tmp1))
        p_change = get_date_return(base_time, max_day_try=15)
        y = np.vstack((y, [p_change]))

    return X1, X2, y

if __name__ == '__main__':
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    training_set_path = os.path.join(this_file_path, "newsdata",)
    file_list = os.listdir(training_set_path,)
    X1 = None
    X2 = None
    y = None
    for idx, f in enumerate(file_list):
        file_path = os.path.join(training_set_path, f)
        X1_tmp, X2_tmp, y_tmp = read_news_and_generate_training_set(file_path)
        X1 = X1_tmp if X1 is None else np.vstack((X1, X1_tmp))
        X2 = X2_tmp if X2 is None else np.vstack((X2, X2_tmp))
        y = y_tmp if y is None else np.vstack((y, y_tmp))

    training_set_path = os.path.join(this_file_path, "testdata",)
    file_list = os.listdir(training_set_path,)
    X1_val = None
    X2_val = None
    y_val = None
    for idx, f in enumerate(file_list):
        file_path = os.path.join(training_set_path, f)
        X1_tmp, X2_tmp, y_tmp = read_news_and_generate_training_set(file_path, year='2016年')
        X1_val = X1_tmp if X1_val is None else np.vstack((X1_val, X1_tmp))
        X2_val = X2_tmp if X2_val is None else np.vstack((X2_val, X2_tmp))
        y_val = y_tmp if y_val is None else np.vstack((y_val, y_tmp))

    np.save('X1', X1)
    np.save('X2', X2)
    np.save('y', y)
    np.save('X1_val', X1_val)
    np.save('X2_val', X2_val)
    np.save('y_val', y_val)

    print X1.shape, X2.shape, y.shape
    print X1_val.shape, X2_val.shape, y_val.shape
