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


class BaseStrategy:

    def __init__(self):

        self.req_samples = 0

    def step(self, current_price_instance, history):

        action = 0
        computed_values = []
        
        price = ComputedValue(current_price_instance.last_price, 'price')
        computed_values.append(price)

        return action, computed_values



class AlertStrategy:

    def __init__(self):

        self.req_samples = 300

        self.factor = 1

        self.delta_percentage = 0.005
        self.samples_between_updates = 500
        self.averaged_samples = 300

        self.bought = False




    def step(self, current_price_instance, history):

        action = 0
        computed_values = []
        average_val = current_price_instance.last_price

                
        if len(history.instances) < self.req_samples:
            self.update_ref(average_val)  
        
        else:
            
            history_computed_values_arrays = history.get_computed_values_arrays(self.req_samples)
            price_history_array = \
                [ar for ar in history_computed_values_arrays if ar.label == 'price'][0].values

            average_val = sum(price_history_array[-self.averaged_samples:]) / self.averaged_samples
            
            state = self.check_alert(average_val)

            if state == 0:
                self.samples_since_update += 1

            else:
                if state > 0:
                    self.update_ref(average_val)
                    action = 0.5

                elif state < 0 :
                    self.update_ref(average_val)
                    action = -0.5
                   
                else:
                    self.samples_since_update += 1

            if self.samples_since_update > self.samples_between_updates:
                self.update_ref(average_val)

        price = ComputedValue(current_price_instance.last_price, 'price')
        computed_values.append(price)

        average = ComputedValue(average_val, 'average_{}_samples'.format(self.averaged_samples))
        computed_values.append(average)

        ref_price = ComputedValue(self.ref_price, 'ref_price')
        computed_values.append(ref_price)

        return action, computed_values


    def update_ref(self, current_price_val):
        
        self.samples_since_update = 0
        self.ref_price = current_price_val
    

    def check_alert(self, current_price_val):

        curr_price = current_price_val

        if curr_price < (1 - self.delta_percentage) * self.ref_price:
            return (curr_price - self.ref_price)/self.ref_price

        if curr_price > (1 + self.delta_percentage) * self.ref_price:
            return (curr_price - self.ref_price)/self.ref_price

        return 0


class CrossingAveragesStrategy:

    def __init__(self):

        self.no_averages = 3
        self.min_avg_samples = 15
        self.max_avg_samples = 500

        self.req_samples = self.max_avg_samples

        self.trend = 0
        
        self.exp = 2
        self.remember_rate = 0.8

        self.max_trend = sum(self.exp ** n for n in range(self.no_averages + 1))

    
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
            
            trend = 0

            for i in range(self.no_averages-1):
                if avgs[i] > avgs[i+1]:
                    trend += i + 1
                    
                    break
                if avgs[i] < avgs[i+1]:
                    trend -= i + 1
                    break
        
            action = trend / sum(range(self.no_averages-1))
            '''
            if abs(self.trend) < 1 and abs(self.trend) > 0:
                action = self.trend
            '''
        computed_values = []
        price = ComputedValue(current_price_instance.last_price, 'price')
        computed_values.append(price)

        #trend = ComputedValue(self.trend * current_price_instance.last_price , 'trend')
        #computed_values.append(trend)

        for i in range(self.no_averages):
            averaged_samples = int(i * (self.max_avg_samples - self.min_avg_samples) / (self.no_averages - 1) + self.min_avg_samples)
            average = ComputedValue(avgs[i], 'average_{}_samples'.format(averaged_samples))
            computed_values.append(average)

        return action, computed_values



ChosenStrategy = CrossingAveragesStrategy