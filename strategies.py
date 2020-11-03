
import numpy as np

class BasicStrategy:
    
    def __init__(self):
        self.no_samples = 30
    
    def get_choice(self, coin1_balance, coin2_balance, price_data, filtered_price_data, gradient_price_data):
        
        if np.random.uniform() > 0.995:
            return np.random.uniform()
        
        if np.random.uniform() > 0.995:
            return -np.random.uniform()
        
        return 0
    

ChosenStrategy = BasicStrategy