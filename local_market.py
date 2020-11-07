#
# https://python-binance.readthedocs.io/en/latest/general.html
# 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from binance.client import Client
import time, os, yaml, json
import numpy as np
from datetime import datetime
from utils import price_instance, get_price_instance, get_prices_from_data
from data_processing import soft_filter, hard_filter, \
    first_grad, second_grad, decomp_grad

# | Binance API Credentials are taken from a separate .yaml file
parameters_file = 'parameters.yaml'


class LocalMarket:

    def __init__(self, coin1, coin2, testing = True, api_wait_time = 5):

        # | Reading parameter file
        
        with open(parameters_file) as file:
            params = yaml.full_load(file)
        api_key = params.get('api_key')
        api_secret = params.get('api_secret')
        
        self.price_data_file = params.get('price_data_file')

        # | Initialising Binance API Client
       
        self.client = Client(
            api_key=api_key,
            api_secret=api_secret
            )
        status = self.client.get_system_status()
        if status.get('status') == 0:
            print('Succesful API Client Initialisation',flush=True)
        else:
            print('Unsuccesful API CLient Initialisation.',flush=True)
        
        # | Initialising class variables
        
        self.testing = testing
        self.coin1 = coin1
        self.coin2 = coin2

        self.coin1_balance = 1
        self.coin2_balance = 1

        self.api_wait_time = api_wait_time
        '''
            self.api_wait_time
            Time (in seconds) between consecutive api calls.
            Minimum value allowed by Binance: 0.05 (20 calls / second)
        '''
        self.price_data = []
        self.loaded_data = None

        if self.testing is True:
            '''
                When in test mode, the price is loaded from the price_data_file,
                so that testing can be done instantaneously
            '''
            if os.path.exists(self.price_data_file):
                with open(self.price_data_file) as json_file:
                    self.loaded_data = json.load(json_file)
        
        else:
            ''' 
                When in live mode, past data is deleted and new data is saved instead.
            '''
            if os.path.exists(self.price_data_file):
                os.remove(self.price_data_file)


    def set_test_values(self,  coin1_balance = 1, coin2_balance = 1, trade_fee = None):
        '''
            Allows for configurating costum testing scenarios.
        '''
        if self.testing is True:
            self.coin1_balance = coin1_balance
            self.coin2_balance = coin2_balance
            if trade_fee is None:
                self.trade_fee = self.client.get_trade_fee(symbol=self.coin1+self.coin2).get('tradeFee')[0].get('maker')
            else:
                self.trade_fee = trade_fee
    

    def get_balance(self):
        self.update_balance()
        return self.coin1_balance, self.coin2_balance


    def update_balance(self):
        
        if self.testing is True:
            pass
        else:
            self.coin1_balance = float(self.client.get_asset_balance(asset=self.coin1)['free'])
            self.coin2_balance = float(self.client.get_asset_balance(asset=self.coin2)['free'])


    def get_price_data(self, samples = 3):
        self.update_price()
        '''
            Return last n samples from available data. 
            Return format bot in packed and unpacked format for ease of use.
        '''
        if len(self.price_data) > samples:
            last_price_data = self.price_data[-samples:]
            return last_price_data
        else:
            return None


    def update_price(self):
        
        if self.testing is True:
            if self.loaded_data is None: 
                '''
                    If there is no local saved price_data, 
                    a random value for the price is generated,
                    in the interval [0,1] with a variation of 
                    max_var.
                '''
                if len(self.price_data) == 0:
                    instance = price_instance(np.random.uniform())
                else:
                    max_var = 0.001
                    d_price = np.random.uniform(-max_var,max_var)
                    last_var, _ = get_price_instance(self.price_data[-1])
                    if last_var < 0:
                        d_price = abs(d_price)
                    if last_var > 1:
                        d_price = -abs(d_price)
                    instance = price_instance(last_var + d_price)
                self.price_data.append(instance)
            else:
                if len(self.loaded_data) == 0:
                    ''' 
                        If the number of recorded samples is not enough, 
                        they are cycled through again.
                    '''
                    self.loaded_data.extend(self.price_data[::-1])
                instance = self.loaded_data.pop(0)
                self.price_data.append(instance)     
        else:
            '''
                If not in test mode, the price is read from the API
            '''
            live_price = float(self.client.get_ticker(symbol = self.coin1+self.coin2)['lastPrice'])
            instance = price_instance(live_price)
            self.price_data.append(instance)

            if len(self.price_data) % 10 == 0:
                '''
                    Periodically write data in file.
                '''
                with open(self.price_data_file, 'w') as json_file:
                    json.dump(self.price_data, json_file)
    

    def buy_coin1(self, coin1_amount):
        
        price12, _ = get_price_instance(self.price_data[-1])
        coin2_amount = coin1_amount * price12

        if coin2_amount > self.coin2_balance:
            print('\nNot enough {} (needed {}, available {})'.format(self.coin2, coin2_amount, self.coin2_balance),flush=True)
            return 0
        else: 
            if self.testing is True:
                self.coin2_balance -= coin2_amount
                self.coin1_balance += (1 - self.trade_fee) * coin1_amount
                #self.print_wallet()
                return 1
            else:
                try:                 
                    order = client.order_market_buy(
                    symbol=self.coin1+self.coin2,
                    quantity=coin1_amount)
                    self.order_history.append(order)
                    self.action_history.append([self.price12_history[-1], coin1_amount, 0])
                    print('\nBought {} {}.'.format(coin1_amount, self.coin1),flush=True) 
                    #self.print_wallet()
                    return 1
                except:
                    print('\nCould not buy {} {}.'.format(coin1_amount, self.coin1),flush=True)
                    return 0
            

    def sell_coin1(self, coin1_amount):

        price12, _ = get_price_instance(self.price_data[-1])
        coin2_amount = coin1_amount * price12

        if coin1_amount > self.coin1_balance:
            print('\nNot enough {} (needed {}, available {})'.format(self.coin1, coin1_amount, self.coin1_balance),flush=True)
            return 0
        else: 
            if self.testing is True:
                self.coin1_balance -= coin1_amount
                self.coin2_balance += (1 - self.trade_fee) * coin2_amount
                #self.print_wallet()
                return 1
            else:
                try:                 
                    order = client.order_market_sell(
                    symbol=self.coin1+self.coin2,
                    quantity=coin1_amount)
                    self.order_history.append(order)
                    self.action_history.append([self.price12_history[-1], 0, coin1_amount])
                    print('\nSold {} {}.'.format(self.coin1, coin1_amount),flush=True)
                    #self.print_wallet()
                    return 1
                except:
                    print('\nCould not buy {} {}.'.format(self.coin1, coin1_amount,flush=True))
                    return 0


    def estimate_wallet_value_coin1(self):
        if self.testing is False:
            self.update_price()
            self.update_balance()
        if len(self.price_data) == 0:
            self.update_price()
        price, _ = get_price_instance(self.price_data[-1])
        coin2_balance_conv = self.coin2_balance / price
        estimated_value = self.coin1_balance + coin2_balance_conv
        return estimated_value
    

    def print_wallet(self):

        if self.testing is False:
            self.update_balance()
        print('Wallet: \n{}: {} \n{}: {}\nEstimated value: {} {}'.format(
            self.coin1, self.coin1_balance, self.coin2, self.coin2_balance, 
            self.estimate_wallet_value_coin1(), self.coin1),
            flush=True)


    def record_data(self, record_time=None, samples=None):
        
        self.testing = False

        if not record_time is None:
            samples = int(record_time / self.api_wait_time)
        
        if not samples is None:

            if os.path.exists(self.price_data_file):
                os.remove(self.price_data_file)
            self.price_data = []

            print('Start recording data for {} samples.'.format(samples), flush=True)

            for _ in tqdm(range(samples)):
                self.update_price()
                time.sleep(self.api_wait_time)



class MarketAnalyser:
    
    def __init__(self):

        # | Reading parameter file
        
        with open(parameters_file) as file:
            params = yaml.full_load(file)
    
        
        self.price_data_file = params.get('price_data_file')

        self.price12 = []

        self.load_data()
        self.plot_data()


    def load_data(self):

        if os.path.exists(self.price_data_file):
            with open(self.price_data_file) as json_file:
                data = json.load(json_file)
            for instance in data:
                price12, time = get_price_instance(instance)
                self.price12.append(price12)


    def plot_data(self):

        # | Plot the first gradient of the data
        grad1 = first_grad(self.price12)
        pos_grad1, neg_grad1 = decomp_grad(self.price12, grad1)
        plt.plot(pos_grad1, linewidth = 3, color = 'lightgreen')
        plt.plot(neg_grad1, linewidth = 3, color = 'lightcoral')

        # | Plot the first gradient of the filtered data
        grad1 = first_grad(soft_filter(self.price12))
        pos_grad1, neg_grad1 = decomp_grad(soft_filter(self.price12), grad1)
        plt.plot(pos_grad1, linewidth = 1, color = 'green')
        plt.plot(neg_grad1, linewidth = 1, color = 'firebrick')

        # | Plot the data with different levels of filtering (in different shades)
        plt.plot(self.price12, linewidth=3, color = 'lightsteelblue')
        plt.plot(soft_filter(self.price12), linewidth=2, linestyle = '-', color = 'royalblue')
        plt.plot(hard_filter(self.price12), linewidth=1, linestyle = '-', color = 'navy')

        plt.grid(True)

        plt.show()
           

if __name__ == '__main__' :

    def gather_data(hours_to_run, minutes_to_run):
    
        local_market = LocalMarket('BNB', 'BTC', testing=False)
        local_market.record_data(record_time = hours_to_run * 3600 + minutes_to_run * 60)

    hours = 8
    minutes = 0

    gather_data(hours,minutes)
    market_analyser = MarketAnalyser()
    
   