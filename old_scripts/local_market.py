#
# https://python-binance.readthedocs.io/en/latest/general.html
# 

import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from binance.client import Client
import time, os, yaml, json
import numpy as np
from datetime import datetime
from data_processing import soft_filter, hard_filter, \
    first_grad, second_grad, decomp_grad

# | Binance API Credentials are taken from a separate .yaml file
parameters_file = 'parameters.yaml'


class PriceInstance:

    def __init__(self, symbol, last_price, volume, weighted_avg_price, price_change_percent):

        self.symbol = symbol
        self.last_price = last_price
        self.volume = volume
        self.weighted_avg_price = weighted_avg_price
        self.price_change_percent = price_change_percent
        self.time = str(datetime.now())


class HistoryInstance:

    def __init__(self, price_instance, coin1_balance, coin2_balance, action):

        self.price_instance = price_instance
        self.coin1_balance = coin1_balance
        self.coin2_balance = coin2_balance
        self.action = action


def plot_price_logs(price_logs_list):

    last_price, volume, weighted_avg_price, price_change_percent = \
        price_logs_to_arrays(price_logs_list)

    scaled_last_price = np.interp(last_price, (last_price.min(), last_price.max()), (0, +1))
    scaled_volume = np.interp(volume, (volume.min(), volume.max()), (0, +1))
    scaled_price_change_percent = np.interp(price_change_percent, (price_change_percent.min(), price_change_percent.max()), (0, +1))
    scaled_weighted_avg_price = np.interp(weighted_avg_price,(weighted_avg_price.min(), weighted_avg_price.max()), (0,+1))
    
    plt.title(price_logs_list[0].symbol + ' ' + str(price_logs_list[0].time))
    plt.plot(scaled_last_price, label = 'scaled last price')
    plt.plot(scaled_volume, label = 'scaled volume')
    #plt.plot(scaled_price_change_percent, label = 'scaled price_change_percent')
    #plt.plot(scaled_weighted_avg_price, label = 'scaled_weighted_avg_price')
    plt.legend()
    plt.grid(True)
    plt.show()


class Market:

    def __init__(self, coin1, coin2):
        self.coin1 = coin1
        self.coin2 = coin2
        self.history = []

    def get_price_instance(self):
        '''
        Get the last price instance, either from a saved file or from the live market.
        '''
        pass

    def buy_coin1(self, coin1_amount):
        '''
        Perform simulated or live buy of coin1_amount.
        '''
        pass

    def sell_coin1(self, coin1_amount):
        '''
        Perform simulated or live sell of coin1_amount.
        '''
        pass
    
    def get_wallet(self):       
        return self.coin1_balance, self.coin2_balance, self.estimate_wallet_value_coin1()


    def estimate_wallet_value_coin1(self): 
        '''
        Estimate the current value of the wallet.
        '''
        price12 = self.history[-1].price_instance.last_price
        coin2_balance_conv = self.coin2_balance / price12
        estimated_value = self.coin1_balance + coin2_balance_conv
        return estimated_value

    def history_to_arrays():
        '''
        Get all values from the history as numpy arrays
        '''

        last_price = np.zeros((len(self.history)))
        volume = np.zeros((len(self.history)))
        weighted_avg_price = np.zeros((len(self.history)))
        price_change_percent = np.zeros((len(self.history)))

        for i in range(len(self.history)):

            last_price[i] = self.history[i].price_instance.last_price
            volume[i] = self.history[i].price_instance.volume
            weighted_avg_price[i] =self.history[i].price_instance.weighted_avg_price
            price_change_percent[i] = self.history[i].price_instance.price_change_percent

        return last_price, volume, weighted_avg_price, price_change_percent 


    def plot_history(price_logs_list):

        last_price, volume, weighted_avg_price, price_change_percent = self.history_to_arrays()
        
        scaled_last_price = np.interp(last_price, (last_price.min(), last_price.max()), (0, +1))
        scaled_volume = np.interp(volume, (volume.min(), volume.max()), (0, +1))
        scaled_price_change_percent = np.interp(price_change_percent, (price_change_percent.min(), price_change_percent.max()), (0, +1))
        scaled_weighted_avg_price = np.interp(weighted_avg_price,(weighted_avg_price.min(), weighted_avg_price.max()), (0,+1))
        
        plt.title(price_logs_list[0].symbol + ' ' + str(price_logs_list[0].time))
        plt.plot(scaled_last_price, label = 'scaled last price')
        plt.plot(scaled_volume, label = 'scaled volume')
        #plt.plot(scaled_price_change_percent, label = 'scaled price_change_percent')
        #plt.plot(scaled_weighted_avg_price, label = 'scaled_weighted_avg_price')
        plt.legend()
        plt.grid(True)
        plt.show()
class TestMarket:

    def __init__(self, coin1, coin2, price_log_file = None, coin1_balance = 10, coin2_balance = 10, trade_fee = 0.01):
        
        self.coin1 = coin1
        self.coin2 = coin2

        self.coin1_balance = coin1_balance
        self.coin2_balance = coin2_balance

        self.trade_fee = trade_fee

        with open(parameters_file) as file:
            params = yaml.full_load(file)

        price_logs_dir = params.get('price_logs_dir')

        if price_log_file is None:
            price_log_file = random.choice(glob.glob(os.path.join(price_logs_dir, coin1 + coin2 + '*')))
        
        if os.path.exists(price_log_file):
            
            self.price_logs = []
            
            with open(price_log_file) as json_file:
                price_logs = json.load(json_file)
            for price_log in price_logs:
                instance = PriceInstance(
                    symbol = price_log['symbol'],
                    last_price = price_log['last_price'],
                    volume = price_log['volume'],
                    weighted_avg_price = price_log['weighted_avg_price'],
                    price_change_percent = price_log['price_change_percent']
                )
                self.price_logs.append(instance)

            print('Succesfully loaded {} price instances from {}'.format(
                len(self.price_logs), os.path.basename(price_log_file)))


    def get_price(self):

        if len(self.price_logs) > 0:
            return self.price_logs.pop(0)
        else:
            return None


    def buy_coin1(self, coin1_amount):
        
        price12 = self.price_logs[0].last_price
        coin2_amount = coin1_amount * price12

        if coin2_amount > self.coin2_balance:
            print('\nNot enough {} (needed {}, available {})'.format(self.coin2, coin2_amount, self.coin2_balance),flush=True)
            return 0
        else: 
            self.coin2_balance -= coin2_amount
            self.coin1_balance += (1 - self.trade_fee) * coin1_amount 
            return 1

            
    def sell_coin1(self, coin1_amount):

        price12 = self.price_logs[0].last_price
        coin2_amount = coin1_amount * price12

        if coin1_amount > self.coin1_balance:
            print('\nNot enough {} (needed {}, available {})'.format(self.coin1, coin1_amount, self.coin1_balance),flush=True)
            return 0
        else: 
            self.coin1_balance -= coin1_amount
            self.coin2_balance += (1 - self.trade_fee) * coin2_amount
            return 1


    def get_wallet(self):

        return self.coin1_balance, self.coin2_balance, self.estimate_wallet_value_coin1()


    def estimate_wallet_value_coin1(self):
        
        price12 = self.price_logs[0].last_price
        coin2_balance_conv = self.coin2_balance / price12
        estimated_value = self.coin1_balance + coin2_balance_conv
        return estimated_value


class LiveMarket:

    def __init__(self, coin1, coin2, api_wait_time = 5):

        # | Reading parameter file
        
        with open(parameters_file) as file:
            params = yaml.full_load(file)
        api_key = params.get('api_key')
        api_secret = params.get('api_secret')


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
                    loaded_data = json.load(json_file)
                    for data in loaded_data:
                        instance = PriceInstance(
                            symbol = data['symbol'],
                            last_price = data['last_price'],
                            volume = data['volume'],
                            weighted_avg_price =data['weighted_avg_price'],
                            price_change_percent = data['price_change_percent'])
                        self.loaded_data_data.append(instance)
        
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
                if len(self.loaded_data) == 0:
                    ''' 
                        If the number of recorded samples is not enough, 
                        they are cycled through again.
                    '''
                    self.loaded_data.extend(self.price_data[::-1])
                self.price_data.append(self.loaded_data.pop(0))
        else:
            '''
                If not in test mode, the price is read from the API
            '''
        
            last_price = float(self.client.get_ticker(symbol = self.coin1+self.coin2)['lastPrice'])
            volume = float(self.client.get_ticker(symbol = self.coin1+self.coin2)['volume'])
            weighted_avg_price = float(self.client.get_ticker(symbol = self.coin1+self.coin2)['weightedAvgPrice'])
            price_change_percent = float(self.client.get_ticker(symbol = self.coin1+self.coin2)['priceChangePercent'])
            
            instance = PriceInstance(
                symbol = self.coin1 + self.coin2,
                last_price = last_price,
                volume = volume,
                weighted_avg_price = weighted_avg_price,
                price_change_percent = price_change_percent
                )
            self.price_data.append(instance.__dict__)

            if len(self.price_data) % 10 == 0:
                '''
                    Periodically write data in file.
                '''
                with open(self.price_data_file, 'w') as json_file:
                    json.dump(self.price_data, json_file)
    

    def buy_coin1(self, coin1_amount):
        
        price12 = self.price_data[-1].last_price
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

        price12 = self.price_data[-1].last_price
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
        price = self.price_data[-1].last_price
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
                loaded_data = json.load(json_file)
            for data in loaded_data:
                instance = PriceInstance(
                    symbol = data['symbol'],
                    last_price = data['last_price'],
                    volume = data['volume'],
                    weighted_avg_price =data['weighted_avg_price'],
                    price_change_percent = data['price_change_percent']
                )
                self.price12.append(instance.last_price)


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

    test_market = TestMarket('BTC', 'USDT')

    plot_price_logs(test_market.price_logs)