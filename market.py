#
# https://python-binance.readthedocs.io/en/latest/general.html
# 

import glob, random, time, os, yaml, json
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from binance.client import Client


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


class TestMarket:

    def __init__(self, coin1, coin2, price_log_file = None, coin1_balance = 0, coin2_balance = 100, trade_fee = 0):
        
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
                self.last_instance = instance
                
            print('Succesfully loaded {} price instances from {}'.format(
                len(self.price_logs), os.path.basename(price_log_file)))


    def get_price_instance(self):
        '''
        Get the last price instance, either from a saved file or from the live market.
        '''
        if len(self.price_logs) > 0:
            self.last_instance = self.price_logs.pop(0)
            return self.last_instance
        else:
            return None


    def buy_coin1(self, percentage):
        
        coin2_amount = percentage * self.coin2_balance 
        coin1_amount = (1 - self.trade_fee) * (coin2_amount / self.price_logs[0].last_price)
        
        if coin1_amount > self.estimate_wallet_value_coin1() * 0.05:
            self.coin2_balance -= coin2_amount
            self.coin1_balance += coin1_amount

            print('Bought {} - {:.2f}% {}'.format(self.coin1, percentage * 100, self.coin2))

            return coin1_amount

        return 0

          
    def sell_coin1(self, percentage):

        coin1_amount = percentage * self.coin1_balance
        coin2_amount = (1 - self.trade_fee) * (coin1_amount * self.price_logs[0].last_price)
        
        if coin1_amount > self.estimate_wallet_value_coin1() * 0.05:

            self.coin1_balance -= coin1_amount
            self.coin2_balance += coin2_amount 

            print('Sold {:.2f}% {}'.format(percentage * 100, self.coin1))

            return -coin1_amount

        return 0
    

    def get_wallet(self):       
        return self.coin1_balance, self.coin2_balance, self.estimate_wallet_value_coin1()


    def estimate_wallet_value_coin1(self): 
        '''
        Estimate the current value of the wallet.
        '''
        price12 = self.last_instance.last_price
        coin2_balance_conv = self.coin2_balance / price12
        estimated_value = self.coin1_balance + coin2_balance_conv
        return estimated_value

   
    def logs_to_arrays(self):
        '''
        Get all values from the history as numpy arrays
        '''
        last_price = np.zeros((len(self.price_logs)))
        volume = np.zeros((len(self.price_logs)))
        weighted_avg_price = np.zeros((len(self.price_logs)))
        price_change_percent = np.zeros((len(self.price_logs)))

        for i in range(len(self.price_logs)):

            last_price[i] = self.price_logs[i].last_price
            volume[i] = self.price_logs[i].volume
            weighted_avg_price[i] = self.price_logs[i].weighted_avg_price
            price_change_percent[i] = self.price_logs[i].price_change_percent

        return last_price, volume, weighted_avg_price, price_change_percent 


    def plot_data(self):

        last_price, volume, weighted_avg_price, price_change_percent = self.logs_to_arrays()
        
        scaled_last_price = np.interp(last_price, (last_price.min(), last_price.max()), (0, +1))
        scaled_volume = np.interp(volume, (volume.min(), volume.max()), (0, +1))
        scaled_price_change_percent = np.interp(price_change_percent, (price_change_percent.min(), price_change_percent.max()), (0, +1))
        scaled_weighted_avg_price = np.interp(weighted_avg_price,(weighted_avg_price.min(), weighted_avg_price.max()), (0,+1))
        
        plt.title(self.price_logs[0].symbol + ' ' + str(self.price_logs[0].time))
        plt.plot(scaled_last_price, label = 'scaled last price')
        plt.plot(scaled_volume, label = 'scaled volume')
        plt.plot(price_change_percent, label = 'scaled price_change_percent')
        plt.plot(scaled_weighted_avg_price, label = 'scaled_weighted_avg_price')
        plt.legend()
        plt.grid(True)
        plt.show()
    

class LiveMarket:

    def __init__(self, coin1, coin2, api_wait_time = 5):

        # | Initialising API Client
        
        with open(parameters_file) as file:
            params = yaml.full_load(file)
        api_key = params.get('api_key')
        api_secret = params.get('api_secret')
       
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
        
        self.coin1 = coin1
        self.coin2 = coin2

        self.coin1_balance = float(self.client.get_asset_balance(asset=self.coin1)['free'])
        self.coin2_balance = float(self.client.get_asset_balance(asset=self.coin2)['free'])

        self.api_wait_time = api_wait_time
        '''
            self.api_wait_time
            Time (in seconds) between consecutive api calls.
            Minimum value allowed by Binance: 0.05 (20 calls / second)
        '''

   
    def get_price_instance(self):
        '''
        Get the last price instance, either from a saved file or from the live market.
        '''
        try:
            data = self.client.get_ticker(symbol = self.coin1 + self.coin2)

            last_price = float(data['lastPrice'])
            volume = float(data['volume'])
            weighted_avg_price = float(data['weightedAvgPrice'])
            price_change_percent = float(data['priceChangePercent'])
                
            instance = PriceInstance(
                symbol = self.coin1 + self.coin2,
                last_price = last_price,
                volume = volume,
                weighted_avg_price = weighted_avg_price,
                price_change_percent = price_change_percent
                )

            return instance
       
        except:
            return None


    def buy_coin1(self, coin1_amount):
        '''
        Perform simulated or live buy of coin1_amount.
        ''' 
        price12 = self.price_data[-1].last_price
        coin2_amount = coin1_amount * price12

        if coin2_amount > self.coin2_balance:
            print('\nNot enough {} (needed {}, available {})'.format(self.coin2, coin2_amount, self.coin2_balance),flush=True)
            return 0
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


    def get_wallet(self):       
        return self.coin1_balance, self.coin2_balance, self.estimate_wallet_value_coin1()


    def estimate_wallet_value_coin1(self): 
        '''
        Estimate the current value of the wallet.
        '''
        price12 = self.get_price_instance().last_price
        coin2_balance_conv = self.coin2_balance / price12
        estimated_value = self.coin1_balance + coin2_balance_conv
        return estimated_value
        

if __name__ == '__main__' :

    
    test_market = TestMarket('BTC', 'USDT')
    test_market.plot_data()
    

    '''
    live_market = LiveMarket('BTC','USDT')
    coin1_balance, coin2_balance, _ = live_market.get_wallet()
    print(coin2_balance)
    '''