
import glob, random, time, os, yaml, json
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from binance.client import Client

from market import TestMarket, LiveMarket
from strategies import ChosenStrategy
from trader_history import ComputedValue, ComputedValuesArray, TraderHistory, HistoryInstance


class Trader:

    def __init__(self, coin1, coin2, testing = True):
               
        self.coin1 = coin1
        self.coin2 = coin2
    
        if testing is True:
            self.market = TestMarket(self.coin1, self.coin2)
            print('\nRunning on a Test Market\n')
        else:
            self.market = LiveMarket(self.coin1, self.coin2)
            print('\nRunning on a LIVE Market\n')

        self.coin1_balance, self.coin2_balance, _ = self.market.get_wallet()

        self.history = TraderHistory()
        self.strategy = ChosenStrategy()


    def run_step(self):
        
        current_price_instance = self.market.get_price_instance()           
       
        if not current_price_instance is None:
            
            action, computed_values = self.strategy.step(current_price_instance, self.history)     
            
            _, _, estimated_value = self.market.get_wallet()

            confirmed_action = 0
            if action > 0:
                confirmed_action = self.market.buy_coin1(action)
            if action < 0:
                confirmed_action = self.market.sell_coin1(-action)

            history_instance = HistoryInstance(
                coin1_balance = self.coin1_balance,
                coin2_balance = self.coin2_balance,
                action = confirmed_action,
                computed_values = computed_values,
                estimated_value = estimated_value
            )

            self.history.add_instance(history_instance)
            
            return 1
        else:
            return None
            
        
    def run(self, samples = None):
        
        coin1_start, coin2_start, estimated_value_start = self.market.get_wallet()


        if samples is None:
            samples = len(self.market.price_logs) - 1
        
        for _ in range(samples-1):
            self.run_step()

        coin1_end, coin2_end, estimated_value_end = self.market.get_wallet()

        print('Wallet change: {:.2f}% ({:.2f} {})'.format(
            100 * (estimated_value_end - estimated_value_start)/estimated_value_start,
            estimated_value_end - estimated_value_start,
            self.coin1))
        
        print('Initially:')
        print('{}{}\n{}{}'.format(coin1_start, self.coin1, coin2_start, self.coin2))
        print('Final:')
        print('{}{}\n{}{}'.format(coin1_end, self.coin1, coin2_end, self.coin2))
        self.history.plot_history()

    '''
    def decide_action(self):
        
        required_samples = self.strategy.no_samples
        
        if len(self.)
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
                """
                    Order has not been succesfull.
                """
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
    '''

if __name__ == '__main__' :


    trader = Trader('BTC', 'USDT', testing=True)
    trader.run(samples = 1000)
