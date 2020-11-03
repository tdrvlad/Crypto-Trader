import numpy as np
import matplotlib.pyplot as plt

import time, yaml
import numpy as np
from scipy.signal import lfilter
from local_market import LocalMarket
from utils import price_instance, get_price_instance, get_prices_from_data
from strategies import ChosenStrategy, BasicStrategy
from tqdm import tqdm
from data_processing import soft_filter, hard_filter, \
    first_grad, second_grad, decomp_grad

class Trader:

    def __init__(self, coin1, coin2, testing = True):
               
        self.testing = testing
        self.coin1 = coin1
        self.coin2 = coin2

        self.market = LocalMarket(self.coin1, self.coin2, testing = testing)
        self.history = History(self.coin1, self.coin2)

        self.coin1_balance, self.coin2_balance = self.market.get_balance()


    def decide_action(self):
        
        strategy = ChosenStrategy()
        price12_data = self.market.get_price_data(strategy.no_samples)
        
        if not price12_data is None:   
            price12 = get_prices_from_data(price12_data)
            filtered_price12 = soft_filter(price12)
            grad1_price12 = first_grad(filtered_price12)
            traded_amount = strategy.get_choice(
                self.coin1_balance, self.coin2_balance, 
                price12, filtered_price12, grad1_price12
                )
            
            self.history.add_instance(
                price12 = price12[-1],
                filtered_price12 = filtered_price12[-1],
                grad1_price12 = grad1_price12[-1],
                balance = self.market.estimate_wallet_value_coin1,
                traded_amount = traded_amount
            )

    
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
        self.history.plot_history()


class History:

    def __init__(self, coin1, coin2):
        self.symbol = coin1 + coin2
        self.price12 = []
        self.filtered_price12 = []
        self.grad1_price12 = []
        self.balance = []
        self.orders = []


    def add_instance(self, price12, filtered_price12, grad1_price12, balance, traded_amount = 0):

        self.price12.append(price12)
        self.filtered_price12.append(filtered_price12)
        self.grad1_price12.append(grad1_price12)
        self.balance.append(balance)
        self.orders.append(traded_amount)
        '''
            A non-zero traded amount means an order was placed. 
            If the amount is positive, the action was a buy.
            If the amount is negative, the action is a sell.
        '''


    def plot_history(self):

        if len(self.price12) != len(self.filtered_price12) or \
            len(self.filtered_price12) != len(self.grad1_price12) or \
            len(self.balance) != len(self.orders):
            print('Corrupted or incomplete data for price and balance.')
        else:
            no_data_points = len(self.price12)
            print('Recorded {} data points.'.format(no_data_points))
            
            # | Plot first gradient of the data
            pos_grad1, neg_grad1 = decomp_grad(self.price12, self.grad1_price12)
            plt.plot(pos_grad1, linewidth = 2, color = 'lightgreen')
            plt.plot(neg_grad1, linewidth = 2, color = 'lightcoral')
            
            # | Plot the data with different levels of filtering (in different shades)
            plt.plot(self.price12, linewidth=3, color = 'grey')
            plt.plot(self.filtered_price12, linewidth=1, color = 'black')
            
            '''
            max_price12 = np.amax(self.price12)
            plt.plot(self.price12, linewidth=1, color='black')

            scaler = max_price12 / np.amax(self.balance)
            
            
            np.amin(data) - np.amax(balance)
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
            '''
            # | Plotting the buy and sell orders.
            '''
                Markers will be blue diamonds for buy and yellow squares for sell.
                The size of the marker will be proportional to the traded amount.
            '''
            max_size = 13
            min_size = 3
            max_amount = np.amax([np.amax(self.orders), np.amin(self.orders)])
            for i in range(len(self.orders)): 
                size = abs(self.orders[i]) * (max_size - min_size) / max_amount + min_size
                if self.orders[i] > 0:
                    plt.plot(i, self.filtered_price12[i], marker='o', color='blue', markersize=size)
                if self.orders[i] < 0:
                    plt.plot(i, self.filtered_price12[i], marker='d', color='magenta', markersize=size)

            plt.grid(True)
            plt.show()




            

if __name__ == '__main__' :


    trader = Trader('BNB', 'BTC', testing=True)
    trader.market.set_test_values(10, 0.1, 0)
    trader.run(3500)


