from binance.client import Client

api_key = '3isbCoakl447UGEElObmz0tffpfQloev3LMjlbyw7vBaUPEINfpVdXwY4f8px98J'
api_secret = 'wQgMCqEE9K2DD4aWnn8YkHy1zzr61sEuPzjZgqFA7ovf4z631eezl2utnZyT8uIQ'

client = Client(api_key, api_secret)

class simWallet:

    def __init__(self):

        self.bnb_balance = float(client.get_asset_balance(asset='BNB')['free'])
        self.btc_balance = float(client.get_asset_balance(asset='BTC')['free'])

        #self.fee = client.get_trade_fee(symbol = 'BNBBTC')['maker']
        self.fee = 0


    def update_price(self):

        self.bnb_btc = float(client.get_ticker(symbol = 'BNBBTC')['lastPrice'])

        return self.bnb_btc

    def buy_bnb(self, amount):
        
        self.update_price()
        btc_amount = amount * self.bnb_btc

        if btc_amount > self.btc_balance:
            print('\nNot enough BTC (needed {}, available {})'.format(btc_amount, self.btc_balance))

        else:
            self.btc_balance -= btc_amount
            self.bnb_balance += (1 - self.fee) * amount

            print('\nBought {} BNB.'.format(amount))
            self.print_wallet()


    def sell_bnb(self, amount):

        self.update_price()
        btc_amount = amount * self.bnb_btc

        if amount > self.bnb_balance:
            print('\nNot enough BNB (needed {}, available {})'.format(amount, self.bnb_balance))
        
        else:
            self.bnb_balance -= amount
            self.btc_balance += (1 - self.fee) * btc_amount
            
            print('\nSold {} BNB.'.format(amount))
            self.print_wallet()


    def print_wallet(self):

        print('Wallet: {} BNB     {} BTC'.format(self.bnb_balance, self.btc_balance))

    
    def estimate_total(self):

        self.update_price()
        btc_amount = self.bnb_balance * self.bnb_btc

        estimated_value = self.btc_balance + btc_amount
       
        return estimated_value


if __name__ == '__main__' :

    wallet = simWallet()


    wallet.print_wallet()
    wallet.estimate_total()

    wallet.sell_bnb(2)
    wallet.buy_bnb(1)
    wallet.sell_bnb(2)
    wallet.buy_bnb(1)

    wallet.estimate_total()
