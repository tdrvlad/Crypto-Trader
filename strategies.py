import numpy as np
from data_processing import first_grad, second_grad, hard_filter, soft_filter
import os,random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, LSTM, LayerNormalization
from keras.optimizers import Adam
import tensorflow as tf

from collections import deque

'''
    Unofficial solving the problem of EageVariableNameReuse raised by GPU usage.
'''
os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

class DQN:
    
    '''
        https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    '''
    
    def __init__(self, model_name, no_inputs, no_outputs, batch_size = 16):
        
        self.model_name = model_name
        
        self.memory  = deque(maxlen=2000)
        
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.batch_size = batch_size

        if os.path.exists(self.model_name):
            print('\nFound model {} locally.'.format(self.model_name))
            model = load_model(self.model_name)
            no_inputs = model.input.get_shape()[1]
            no_outputs = model.output.get_shape()[1]
            if no_inputs != self.no_inputs or no_outputs != self.no_outputs:
                print('Local Model incompatible.')
        else:
            print('Creating model {}.'.format(self.model_name))

        self.model        = self.get_model()
        self.target_model = self.get_model()

        self.keras_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.model_name,'logs')),]

    def get_model(self):
        
        if os.path.exists(self.model_name):
            model = load_model(self.model_name)
            no_inputs = model.input.get_shape()[1]
            no_outputs = model.output.get_shape()[1]
            if no_inputs != self.no_inputs or no_outputs != self.no_outputs:
                os.rmdir(self.model_name)
            else:
                return model
        else:
            return self.create_model()


    def create_model(self):

        model   = Sequential()
        model.add(Input(self.no_inputs, name = 'Input'))
        model.add(Dense(self.no_inputs * 2, activation="relu"))
        model.add(Dense(self.no_inputs * self.no_outputs, activation="relu"))
        #model.add(Dense(self.no_outputs * 2, activation="relu"))
        model.add(Dense(self.no_outputs))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            '''
                Generate a random action
            '''
            action =  np.random.uniform(0, 1, self.no_outputs)
        else:
            '''
                Get an action from model's decision
            '''
            action = self.model.predict(state)[0]
        return action


    def bool_act(self, state):
        action = self.act(state)
        return np.argmax(action)


    def remember(self, state, action, reward, new_state):
        self.memory.append([state, action, reward, new_state])


    def replay(self):
        if len(self.memory) < self.batch_size: 
            return

        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state = sample
            target = self.target_model.predict(state)

            Q_future = max(self.target_model.predict(new_state)[0])
            target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        self.model.save(self.model_name)


class DQNStrategy:

    def __init__(self):

        self.model_name = 'dqn_model'

        print('\nUsing Trading Strategy DQN\n')

        self.no_samples = 30
        self.last_state = None
        self.last_action = None
        self.last_balance = None

        self.no_inputs = 4
        '''
            Inputs of the model: 
                - trend_gradient
                - trend_second_gradient
                - lookout_gradient
                - trend_price / lookout_price
        '''
        self.no_outputs = 3
        '''
            Outputs of the model:
                - buy
                - sell
                - stall
        '''
        self.config_dqn_model()


    def state(self, trend_grad, trend_sec_grad, lookout_grad, trend_lookout_price):

        return np.array([
            trend_grad,
            trend_sec_grad,
            lookout_grad,
            trend_lookout_price
            ]).reshape(1, self.no_inputs)


    def config_dqn_model(self):
        
            self.model = DQN(
                model_name = self.model_name,
                no_inputs = self.no_inputs,
                no_outputs = self.no_outputs
            )
    
    def save_model(self):
        self.model.save_model()

    def get_choice(self, coin1_balance, coin2_balance, price12, filtered_price12, gradient_price12):

        # | Estimate Wallet Value
        coin2_balance_conv = coin2_balance / price12[-1]
        ballance = coin1_balance + coin2_balance_conv

        # | Compute relevante values
        trend_samples = int(np.sqrt(self.no_samples))
        
        trend_gradient = np.sum(soft_filter(gradient_price12)[-trend_samples:])
        trend_second_gradient = np.sum(first_grad(hard_filter(gradient_price12))[-trend_samples])
        lookout_gradient = np.sum(soft_filter(gradient_price12))
        trend_lookout_price = np.average(price12[-trend_samples]) / np.average(price12)

        self.new_state = self.state(trend_gradient, trend_second_gradient, lookout_gradient, trend_lookout_price)
        if not self.last_state is None and not self.last_action is None and not self.last_balance is None:
            self.model.remember(
                state = self.last_state,
                action = np.argmax(self.last_action),
                reward = (ballance - self.last_balance) * np.max(self.last_action),
                new_state = self.new_state
            )

            '''
                A reward is associated with a past action to indicate wether its outcome was beneficial or not.
                The action_space is an array [buy, sell, hold] with each value indicating the confidence for the resp. action.
                The reward will be computed as the profit weighted by the confidence.
            '''

            if np.random.uniform() > 0.8:
                self.model.replay()
                self.model.target_train()

        self.action = self.model.act(self.new_state)
        self.action[self.action<0] = 0

        self.last_state = self.new_state
        self.last_action = self.action
        self.last_balance = ballance

        print('\nAction: {}'.format(self.action),flush=True)
        buy_decision = self.action[0] / np.sum(self.action)
        sell_decision = self.action[1] / np.sum(self.action)
        hold_decision = self.action[2] / np.sum(self.action)

        decision = [buy_decision, sell_decision, hold_decision]
        print('Decision: {}'.format(decision))
        print('Balance: C1: {}, C2: {} ({} in C1)'.format(coin1_balance, coin2_balance, coin2_balance_conv))
        if np.argmax(decision) == 0:
            '''
                BUY
            '''
            print('Buy {} C1'.format(buy_decision * coin2_balance_conv))
            return buy_decision * coin2_balance_conv
        
        if np.argmax(decision) == 1:
            '''
                SELL
            '''
            print('Sell {} C1'.format(sell_decision * coin1_balance))
            return -sell_decision * coin1_balance
        return 0


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
        self.factor_history = []

    def get_choice(self, coin1_balance, coin2_balance, price12, filtered_price12, gradient_price12):
        
        coin2_balance_conv = coin2_balance / price12[-1]

        trend_samples = int(np.sqrt(self.no_samples))
        
        trend_gradient = np.sum(soft_filter(gradient_price12)[-trend_samples:])
        trend_second_gradient = np.sum(first_grad(hard_filter(gradient_price12))[-trend_samples])
        lookout_gradient = np.sum(soft_filter(gradient_price12))
        reference_gradient = abs(np.average(gradient_price12))
        
        '''
            Buy condition:
            - trend_gradient is positive (price is increasing)
            - trend_gradient is higher than lookout gradient (trying to modelate a dip)
            Amount:
            - proportional to trend_gradient - reference_gradient (we look for sudden increases in trend compared to a normal state)
            - liniar to trend_second_gradient (we look for the acceleration of the increase in price)
        '''

        factor = abs(trend_gradient) + abs(reference_gradient)
        base = 2
        if trend_second_gradient > 0:
            factor += base ** trend_gradient
        else:
            factor -= base ** trend_gradient
        self.factor_history.append(factor)
        
        # | Scaling the factor in [0,1] range
        max_fact = np.amax(self.factor_history)
        min_fact = np.amin(self.factor_history)
        if max_fact != min_fact:
            scaled_factor = (factor - min_fact) * 0.1 / (max_fact - min_fact)
            scaled_var = ((max_fact - min_fact) - min_fact) * 0.1 / (max_fact - min_fact)
        else:
            scaled_factor = 0.1
            scaled_var = 0.1

        if scaled_factor > scaled_var / 2 and len(self.factor_history) > self.no_samples ** 2:
            if trend_gradient > 0 and trend_gradient > lookout_gradient:
                amount = coin2_balance_conv * scaled_factor
                return amount

            if trend_gradient < 0 and trend_gradient < lookout_gradient:
                amount = coin1_balance * scaled_factor
                return -amount
    
        return 0
    

ChosenStrategy = DQNStrategy