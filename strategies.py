import numpy as np
from data_processing import first_grad, second_grad

class DummyStrategy:
    
    def __init__(self):
        self.no_samples = 20
    
    def get_choice(self, coin1_balance, coin2_balance, price12, filtered_price, gradient_price12):
        
        if np.random.uniform() > 0.995:
            return np.random.uniform()
        
        if np.random.uniform() > 0.995:
            return -np.random.uniform()
        
        return 0

class BasicGradStrategy:
    
    def __init__(self):
        self.no_samples = 30
    
    def get_choice(self, coin1_balance, coin2_balance, price12, filtered_price12, gradient_price12):
        
        coin2_balance_conv = coin2_balance / price12[-1]

        trend_samples = int(np.sqrt(self.no_samples))
        
        trend_gradient = np.sum(gradient_price12[-trend_samples:])
        trend_second_gradient = np.sum(first_grad(gradient_price12)[-trend_samples])
        lookout_gradient = np.sum(gradient_price12)
        reference_gradient = abs(np.average(gradient_price12))
        
        scaler = np.average(price12) / np.amax(gradient_price12)
        '''
            Buy condition:
            - trend_gradient is positive (price is increasing)
            - trend_gradient is higher than lookout gradient (trying to modelate a dip)
            Amount:
            - proportional to trend_gradient - reference_gradient (we look for sudden increases in trend compared to a normal state)
            - liniar to trend_second_gradient (we look for the acceleration of the increase in price)
        '''
        
        if trend_gradient > 0 and trend_gradient > lookout_gradient:
            factor = ((trend_gradient - reference_gradient) + trend_second_gradient) * scaler
            print('Factor buy: {}'.format(factor))
            if factor > 1:
                factor = 1
            amount = coin2_balance_conv * factor
            if factor > 0.01:
                amount = coin1_balance * factor
                return amount

        if trend_gradient < 0 and trend_gradient < lookout_gradient:
            factor = abs((trend_gradient - reference_gradient) + trend_second_gradient) * scaler
            print('Factor sell: {}'.format(factor))
            if factor > 1:
                factor = 1
            if factor > 0.01:
                amount = coin1_balance * factor
                return -amount
    
        return 0
    


ChosenStrategy = BasicGradStrategy