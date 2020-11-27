



import glob, random, time, os, yaml, json
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from binance.client import Client

from market import TestMarket, LiveMarket


class ComputedValue:
    def __init__(self, value, label):   
        self.value = value
        self.label = label

class ComputedValuesArray:
    def __init__(self, values, label):
        self.values = values
        self.label = label

class HistoryInstance:
    def __init__(self, coin1_balance, coin2_balance, estimated_value, action, computed_values):
        self.coin1_balance = coin1_balance
        self.coin2_balance = coin2_balance
        self.estimated_value = estimated_value
        self.computed_values = computed_values
        self.action = action
        self.time = str(datetime.now())


class TraderHistory:

    def __init__(self):
        self.instances = []


    def add_instance(self, history_instance):
        self.instances.append(history_instance)


    def get_computed_values_arrays(self, samples):
        no_computed_values = len(self.instances[-1].computed_values)
        
        labels = []
        for comp_value in self.instances[-1].computed_values:
            labels.append(comp_value.label)

        computed_values = np.zeros((samples, no_computed_values))

        for i in range(samples):
            history_instance = self.instances[-samples + i]
            if len(history_instance.computed_values) == no_computed_values:
                for j in range(no_computed_values):
                    computed_values[i,j] = history_instance.computed_values[j].value

        computed_values_arrays = []
        for j in range(no_computed_values):
            array = ComputedValuesArray(computed_values[:,j], labels[j])
            computed_values_arrays.append(array)

        return computed_values_arrays


    def plot_history(self):
        
        samples = len(self.instances)
        computed_values_arrays = self.get_computed_values_arrays(samples)

        print('Found {} history instances.'.format(samples))
        print('History instances have {} computed value(s).'.format(len(computed_values_arrays)))
        
        plt.figure()
        for array in computed_values_arrays:

            values = array.values
            scaled_values = np.interp(values, (values.min(), values.max()), (0,1))
            plt.plot(scaled_values, label = array.label)

        plt.legend()
        plt.grid(True)
        plt.show()




