
import numpy as np

class BasicStrategy:
    
    def __init__(self):
        self.no_samples = 5
    
    def get_choice(self, coin1_balance, coin2_balance, price_data):
        
        if np.random.uniform() > 0.99:
            return 1, 0.1
        
        if np.random.uniform() > 0.99:
            return -1, 0.1
        
        return None, None
    

ChosenStrategy = BasicStrategy