# -*- coding: UTF-8 -*-
from __future__ import division
from keras.engine.training import Model
from keras.layers import Input,LSTM, Dense, Dropout, Merge
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from fa.model import FinancialNewsAnalysisModel
from ta.multi_task_model import FinancialTimeSeriesAnalysisModel

import numpy as np
import os
from keras.utils.visualize_util import plot
__author__ = 'shenxiangxiang@gmail.com'
nb_hidden_units = 200
dropout = 0.3
l2_norm_alpha = 0.001


class CombinedAnalysisModel(object):
    model = None

    def __init__(self, dim_input_x1, time_step_x1, dim_input_x2, time_step_x2, batch_size=1,
                 model_path=None, fa_model_path=None, ta_model_path=None):
        self.model_path = model_path
        self.fa_model_path = fa_model_path
        self.ta_model_path = ta_model_path
        self.batch_size = batch_size
        self.dim_input_x1 = dim_input_x1
        self.time_step_x1 = time_step_x1
        self.dim_input_x2 = dim_input_x2
        self.time_step_x2 = time_step_x2
        self.build()
        self.weight_loaded = False
        self.load_weights()

    def build(self):
        news_input = Input(shape=(self.time_step_x1, self.dim_input_x1), name='x1')
        financial_time_series_input = Input(shape=(self.time_step_x2, self.dim_input_x2), name='x2')
        lstm = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                    W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha),
                    activation='tanh', name='h1')
        bi_lstm = Bidirectional(lstm,
                                input_shape=(self.time_step_x1, self.dim_input_x1), merge_mode='concat', name='h1')
        h1 = bi_lstm(news_input)

        lstm_layer_1 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                            W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                            return_sequences=True, name='lstm_layer1')
        lstm_layer_23 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                             W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                             return_sequences=False, name='lstm_layer2_loss3')
        h2_layer_1 = lstm_layer_1(financial_time_series_input)
        h2_layer_2 = lstm_layer_23(h2_layer_1)
        h_3 = Merge(mode='concat', name='h3')([h1, h2_layer_2])
        h_4 = Dense(nb_hidden_units, name='h4')(h_3)
        prediction = Dense(1, name='y3')(h_4)
        self.model = Model(input=[news_input, financial_time_series_input],
                           output=prediction,
                           name='combined model for financial analysis')
        plot(self.model, to_file='model.png')

    def reset(self):
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_status()

    def compile_model(self, lr=0.0001, loss_weights=0.1):
        optimizer = Adam(lr=lr)
        loss = 'mse'
        # loss = custom_objective
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_model(self, X, y, X_val=None, y_val=None, epoch=500):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        if X_val is None:
            self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=epoch, validation_split=0.2,
                           shuffle=True, callbacks=[early_stopping])
        else:
            self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=epoch, validation_data=(X_val, y_val),
                           shuffle=True, callbacks=[early_stopping])

    def save(self):
        self.model.save_weights(self.model_path, overwrite=True)

    def load_weights(self):
        if self.model_path is not None and os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            self.weight_loaded = True
        if self.ta_model_path is not None and os.path.exists(self.ta_model_path):
            self.model.load_weights(self.ta_model_path, by_name=True)
        if self.fa_model_path is not None and os.path.exists(self.fa_model_path):
            self.model.load_weights(self.fa_model_path, by_name=True)

    def print_weights(self, weights=None, detail=False):
        weights = weights or self.model.get_weights()
        for w in weights:
            print("w%s: sum(w)=%s, ave(w)=%s" % (w.shape, np.sum(w), np.average(w)))
        if detail:
            for w in weights:
                print("%s: %s" % (w.shape, w))

    def model_eval(self, X, y):
        y_hat = self.model.predict(X, batch_size=1)
        count_true = 0
        count_all = y.shape[0]
        for i in range(y.shape[0]):
            count_true = count_true + 1 if y[i,0]*y_hat[i,0]>0 else count_true
            print y[i,0],y_hat[i,0]
        print count_all,count_true

if __name__ == '__main__':
    model = CombinedAnalysisModel(100, 200, 5, 400, model_path='ca.model.weights',
                                  fa_model_path="fa\\fa.model.weights", ta_model_path="ta\\multask_ta.model.weights")
