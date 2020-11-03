from datetime import datetime

def price_instance(price):
    '''
        Pack the instant information as a dictionary.
    '''
    instance = {
        'price' : price,
        'time' : str(datetime.now())
    }
    return instance


def get_price_instance(instance):
    '''
        Unpack the instant information from a dictionary.
    '''
    price = instance.get('price')
    time = instance.get('time')
    return price, time


def get_prices_from_data(data):
    '''
        Unpack prices froma list of instances.
    '''
    prices = []
    for instance in data:
        price, _ = get_price_instance(instance)
        prices.append(price)
    return prices
