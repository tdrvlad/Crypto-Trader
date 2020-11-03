import numpy as np
import matplotlib.pyplot as plt

import time, yaml
import numpy as np
from scipy.signal import lfilter
from local_market import LocalMarket
from utils import price_instance, get_price_instance, get_prices_from_data
from strategies import ChosenStrategy, BasicStrategy
from tqdm import tqdm

class Trader:

    def __init__(self, coin1, coin2, testing = True):
               
        self.testing = testing
        self.coin1 = coin1
        self.coin2 = coin2

        self.market = LocalMarket(self.coin1, self.coin2, testing = testing)

        self.history = History(self.coin1, self.coin2)

        self.coin1_balance, self.coin2_balance = self.market.get_balance()


    def decide_action(self):
        
        strategy = BasicStrategy()
        price12_data = self.market.get_price_data(strategy.no_samples)
        
        if not price12_data is None:   
            price12 = get_prices_from_data(price12_data)
            action, amount = strategy.get_choice(self.coin1_balance, self.coin2_balance, price12)
            
            moment = Moment(
                self.coin1, self.coin2, 
                price12[-1],
                self.market.estimate_wallet_value_coin1()
                )

            if action == 1:     
                if self.market.buy_coin1(amount) == 1:
                    moment.add_decision('buy')
            
            if action == -1:
                if self.market.sell_coin1(amount) == 1:
                    moment.add_decision('sell')
    
            self.history.add_moment(moment=moment)

    
    def run(self, samples, wait_time = 1):
        
        if self.testing is True:
            print('Running in test mode.')
            wait_time = 0
            if not self.market.loaded_data is None:
                samples = len(self.market.loaded_data)
                print('Found local data: {} samples.'.format(samples))
            else:
                print('No local data found. Will generate {} random samples.'.format(samples))
        else:
            print('Running in live mode.')
            print('Will run for {} samples.'.format(samples))
        self.market.print_wallet()
        for _ in tqdm(range(samples)):
            #print('{} price: {}'.format(self.coin1+self.coin2, self.market.price_data[-1].get('price')), flush=True)
            self.decide_action()
            time.sleep(wait_time)
        self.market.print_wallet()
        self.history.print_history()
        self.history.plot_history()


class Moment:

    def __init__(self, coin1, coin2, price12, instant_balance):
      
        self.symbol = coin1 + coin2
        self.price12 = price12
        self.instant_balance = instant_balance
        
        # /data is an array of dictionaries refering to aditional 
        # information used for decision like first or second order gradients
        self.data = []
        self.buy = False
        self.sell = False
    
    def add_data(self, name, value):
        data_instance = {
            'name' : name,
            'value' : value
        }
        self.data.append(data_instance)
    
    def add_decision(self, action):
        if action == 'buy':
            self.buy = True
        if action == 'sell':
            self.sell = True


class History:

    def __init__(self, coin1, coin2):
        pass

        self.symbol = coin1 + coin2
        self.price12 = []
        self.balance = []
        self.data = {} 
        self.buy_actions = []
        self.sell_actions = []


    def add_moment(self, moment):

        self.price12.append(moment.price12)
        self.balance.append(moment.instant_balance)

        for data_instance in moment.data:
            data_name = data_instance.get('name')
            data_value = data_instance.get('value')
            if data_name not in self.data:
                self.data[data_name] = []
            self.data[data_name].append(data_value)
        
        if moment.buy == True:
            self.buy_actions.append(moment.price12)
        else:
            self.buy_actions.append(np.nan)
        if moment.sell == True:
            self.sell_actions.append(moment.price12)
        else:
            self.sell_actions.append(np.nan)


    def plot_history(self):

        if len(self.price12) != len(self.balance):
            print('Corrupted or incomplete data for price and balance.')
        else:
            no_data_points = len(self.price12)
            print('Recorded {} data points.'.format(no_data_points))
            
            max_price12 = np.amax(self.price12)
            plt.plot(self.price12, linewidth=2, color='black')

            scaler = max_price12 / np.amax(self.balance)
            scaled_balance = [val * scaler for val in self.balance]
            #plt.plot(scaled_balance, linewidth=2, color = 'red')

            # For making the data plots different colors
            no_plots = len(self.data.keys())
            col_var = 1/(no_plots+1)
            col = 0

            for data_type in self.data.keys():
                if len(self.data[data_type]) != no_data_points:
                    print('Corrupted or incomplete data for {}.'.format(data_type))
                else:
                    col += col_var
                    #scaler = max_price12 / np.amax(self.data[data_type])
                    #scaled_data = [val * scaler for val in self.data[data_type]]
                    plt.plot(self.data[data_type], linewidth=1, color=(col,col,col))
                    print('Plotted {}.'.format(data_type))
            
            plt.plot(self.buy_actions, marker='D', color='dodgerblue')
            plt.plot(self.sell_actions, marker='s', color='peru')
            plt.grid(True)
            plt.show()


    def print_history(self):
        if len(self.price12) != len(self.balance):
            print('Corrupted or incomplete data for price and balance.')
            print('{} samples for price; {} samples for balance'.format(len(self.price12),len(self.balance)))
        else:
            no_data_points = len(self.price12)
            print('Recorder {} samples.'.format(no_data_points))
        
            for data_type in self.data.keys():
                if len(self.data[data_type]) != no_data_points:
                    print('Corrupted or incomplete data for {}.'.format(data_type))
                else:
                    print('Recorderd data for {}.'.format(data_type))

            

if __name__ == '__main__' :


    trader = Trader('BNB', 'BTC', testing=True)
    trader.market.set_test_values(10, 0.1, 0)
    trader.run(3500)


