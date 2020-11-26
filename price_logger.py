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
from market import PriceInstance

# | Binance API Credentials are taken from a separate .yaml file
parameters_file = 'parameters.yaml'

with open(parameters_file) as file:
    params = yaml.full_load(file)


class PriceLogger:
    
    def __init__(self, coin1, coin2, api_wait_time = 10, max_logs = 30000):
        
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

        self.coin1 = coin1
        self.coin2 = coin2

        self.new_log_file()

        self.api_wait_time = api_wait_time
        self.max_logs = max_logs

        self.run()

    def new_log_file(self):

        self.price_log = []

        price_logs_dir = params.get('price_logs_dir')
        if not os.path.exists(price_logs_dir):
            os.mkdir(price_logs_dir)

        dt = datetime.now()
        timestamp = str(dt.month) + '_' + str(dt.day) + '_' + str(dt.hour)
        log_file_name = self.coin1 + self.coin2 + '__' + timestamp + '.json'
        self.log_file = os.path.join(price_logs_dir, log_file_name)


    def log(self):

        try:
            data = self.client.get_ticker(symbol = self.coin1+self.coin2)

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

            self.price_log.append(instance.__dict__)
       
        except:
            print('Error getting data.')
            
            last_instance = self.price_log[-1]
            self.price_log.append(last_instance)


        if len(self.price_log) % 10 == 0:
            '''
                Periodically write data in file.
            '''
            with open(self.log_file, 'w') as json_file:
                json.dump(self.price_log, json_file)

        if len(self.price_log) > self.max_logs:
            
            with open(self.log_file, 'w') as json_file:
                json.dump(self.price_log, json_file)

            self.new_log_file()


    def run(self):

        print('Started logging price.')

        while(True):
            self.log()
            time.sleep(self.api_wait_time)



if __name__ == '__main__':

    logger = PriceLogger('BTC', 'USDT')