import numpy as np
import matplotlib.pyplot as plt

import time, yaml
import numpy as np
from scipy.signal import lfilter
from local_market import LocalMarket
from strategies import ChosenStrategy
from tqdm import tqdm
from data_processing import soft_filter, hard_filter, \
    first_grad, second_grad, decomp_grad


class HistoryInstance:

    def __init__(self, price_instance, coin1_balance, coin2_balance, action):

        self.price_instance = price_instance
        self.coin1_balance = coin1_balance
        self.coin2_balance = coin2_balance
        self.action = action

        
class Trader:

    def __init__(self, coin1, coin2, testing = True):
               
        self.testing = testing
        self.coin1 = coin1
        self.coin2 = coin2

        self.market = LocalMarket(self.coin1, self.coin2, testing = testing)
        self.history = History(self.coin1, self.coin2)

        self.coin1_balance, self.coin2_balance = self.market.get_balance()
        self.strategy = ChosenStrategy()


    def decide_action(self):
        
        self.coin1_balance, self.coin2_balance = self.market.get_balance()
        price_instance_samples = self.market.get_price_data(self.strategy.no_samples)
        
        if not price_instance_samples is None:   

            traded_amount = self.strategy.get_choice(
                coin1_balance = self.coin1_balance, 
                coin2_balance = self.coin2_balance, 
                price_instance_samples = price_instance_samples
                )

            resp = None
            if traded_amount > 0:
                resp = self.market.buy_coin1(traded_amount)
            if traded_amount < 0:
                resp = self.market.sell_coin1(abs(traded_amount))
            if resp is None:
                '''
                    Order has not been succesfull.
                '''
                traded_amount = 0
            
            self.history.add_instance(
                price12 = price_instance_samples[-1].last_price,
                instant_balance = self.market.estimate_wallet_value_coin1(),
                traded_amount = traded_amount
            )

    
    def run(self, samples = 100, wait_time = 1):
        
        if self.testing is True:
            print('Running in test mode.')
            wait_time = 0
            if not self.market.loaded_data is None:
                if samples > len(self.market.loaded_data):
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
        #self.strategy.save_model()
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


    def add_instance(self, price12, instant_balance, traded_amount = 0):

        self.price12.append(price12)
        self.balance.append(instant_balance)
        self.orders.append(traded_amount)
        '''
            A non-zero traded amount means an order was placed. 
            If the amount is positive, the action was a buy.
            If the amount is negative, the action is a sell.
        '''


    def plot_history(self):

        if len(self.price12) != len(self.balance) != len(self.orders):
            print('Corrupted or incomplete data for price and balance.')
        else:
            no_data_points = len(self.price12)
            print('Recorded {} data points.'.format(no_data_points))
        
            
            # | Plot the data with different levels of filtering (in different shades)
            plt.plot(self.price12, linewidth=3, color = 'grey')
            
            
            # | Plot the balance total estimate
            #scaler = (np.amax(self.price12) - np.amin(self.price12)) / (np.amax(self.balance) - np.amin(self.balance))
            scaler = np.amax(self.price12) / np.amax(self.balance)
            scaled_balance = np.array(hard_filter(self.balance)) * scaler
            plt.plot(scaled_balance, linewidth=2, linestyle=':', color = 'dodgerblue')


            # | Plot the buy and sell orders.
            '''
                Markers will be blue diamonds for buy and yellow squares for sell.
                The size of the marker will be proportional to the traded amount.
            '''
            max_size = 15
            min_size = 3
            max_amount = max(abs(np.amax(self.orders)), abs(np.amin(self.orders)))
            for i in range(len(self.orders)): 
                size = abs(self.orders[i]) * (max_size - min_size) / max_amount + min_size
                if self.orders[i] > 0:
                    plt.plot(i, self.filtered_price12[i], marker='o', color='blue', markersize=size)
                if self.orders[i] < 0:
                    plt.plot(i, self.filtered_price12[i], marker='d', color='magenta', markersize=size)

            plt.grid(True)
            plt.show()


if __name__ == '__main__' :


    trader = Trader('BTC', 'USDT', testing=True)
    trader.market.set_test_values(10, 0, None)
    trader.run(samples = 1000)


