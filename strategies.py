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

        self.req_samples_compute = 50
        self.req_samples_decide = 100


    def step(self, current_price_instance, history):
        
        if len(history.instances) < self.req_samples:
            price = ComputedValue(current_price_instance.last_price, 'price')
            comp_val_1 = ComputedValue(0, 'random')
        else:
            price = ComputedValue(current_price_instance.last_price, 'price')
            comp_val_1 = ComputedValue(np.random.uniform(), 'random')

            history_computed_values_arrays = history.get_computed_values_arrays(self.req_samples)
            '''
            print('Strategy received {} computed_values of shape {}.'.format(
                len(history_computed_values_arrays),
                history_computed_values_arrays[0].values.shape
            ))
            '''

        computed_values = [price, comp_val_1]
        
           
        return random.choice([0,np.random.uniform()]), computed_values





ChosenStrategy = BasicStrategy