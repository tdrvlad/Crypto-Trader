from sim_wallet import simWallet
from tdigest import TDigest

import time

digest = TDigest()

wallet = simWallet()

wallet.print_wallet()

initial_value = wallet.estimate_total()

print('Initial wallet value is {} BTC.'.format(initial_value), flush=True)

while True:

    current_price = wallet.update_price()
    digest.update(current_price)

    digest_value = digest.percentile(15)

    print('\n\nCurrent BNB/BTC price is {}. Digest value is {}'.format(current_price, digest_value), flush=True)

    if current_price < 0.9 * digest_value:
        wallet.buy_bnb(1)
    
    if current_price > 1.1 * digest_value:
        wallet.sell_bnb(1)

    percent = int(wallet.estimate_total() / initial_value * 100)
    print('\nCurrent wallet value is {}% of initial'.format(percent), flush=True)

    time.sleep(30)