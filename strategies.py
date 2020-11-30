import numpy as np
import os,random
'''
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, LSTM, LayerNormalization
from keras.optimizers import Adam
import tensorflow as tf

from collections import deque

os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
'''

from trader_history import ComputedValue, ComputedValuesArray, HistoryInstance

class RandomStrategy:
    def __init__(self):

        self.req_samples = 50

    def update(self, current_price_instance):
        
        price = ComputedValue(current_price_instance.last_price, 'price')
        comp_val_1 = ComputedValue(np.random.uniform(), 'random')

        computed_values = [price, comp_val_1]
        
        return computed_values

    def get_action(self, computed_values_arrays):

        print('Strategy received {} computed_values of shape {}.'.format(
            len(computed_values_arrays),
            computed_values_arrays[0].values.shape
        ))

        return random.choice([0,np.random.uniform()])


class BasicStrategy:

    def __init__(self):

        self.req_samples = 80

        self.no_averages = 2
        self.min_avg_samples = 20
        self.max_avg_samples = 70

        self.up_trend = False
        self.down_trend = False

    def step(self, current_price_instance, history):
        
        action = 0
        
        if len(history.instances) < self.req_samples:

            avgs = []
            for _ in range(self.no_averages):
                avgs.append(None) 
            
        else:
            price = ComputedValue(current_price_instance.last_price, 'price')
            history_computed_values_arrays = history.get_computed_values_arrays(self.req_samples)
            
            price_history_array = \
                [ar for ar in history_computed_values_arrays if ar.label == 'price'][0].values

            avgs = []
            for i in range(self.no_averages):
                averaged_samples = int(i * (self.max_avg_samples - self.min_avg_samples) / (self.no_averages - 1) + self.min_avg_samples)
                average_val = sum(price_history_array[-averaged_samples:]) / averaged_samples
                avgs.append(average_val) 
            

            if avgs[0] > avgs[1] and self.up_trend != True:
                self.up_trend = True
                self.down_trend = False
                action = 1
            
            if avgs[0] < avgs[1] and self.down_trend != True:
                self.down_trend = True
                self.up_trend = False
                action = -1

        computed_values = []
        price = ComputedValue(current_price_instance.last_price, 'price')
        computed_values.append(price)

        for i in range(self.no_averages):
            averaged_samples = int(i * (self.max_avg_samples - self.min_avg_samples) / (self.no_averages - 1) + self.min_avg_samples)
            average = ComputedValue(avgs[i], 'average_{}_samples'.format(averaged_samples))
            computed_values.append(average)

        return action, computed_values



ChosenStrategy = BasicStrategy