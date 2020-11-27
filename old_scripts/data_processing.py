from scipy.signal import lfilter
import numpy as np

def soft_filter(data):
    return general_filter(data, n = 5)

def hard_filter(data):
    return general_filter(data, n = 20)

def general_filter(data, n = 10):
    '''
        The larger the filtering factor, the smoother the curve.
        The effect f the smoothness if a delay between the 2 graphs.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    '''
    if len(data) > n:
        filtered_data = lfilter(
            b = [1.0 / n] * n,
            a = 1,
            x = data)
        filtered_data[:n] = data[:n]
    else:
        filtered_data = data

    return filtered_data

def first_grad(data):
    return np.gradient(data, 1)

def second_grad(data):
    return np.gradient(data,2)

def decomp_grad(data, grad):
    pos_grad = np.array([x if x > 0 else 0 for x in grad])
    neg_grad = np.array([x if x < 0 else 0 for x in grad])
    scaler = (np.amax(data) - np.amin(data)) / (np.amax(grad) - np.amin(grad))
    scaler *= 0.7
    return pos_grad * scaler + data, neg_grad * scaler + data

