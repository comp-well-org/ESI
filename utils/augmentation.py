import numpy as np

class Jitter(object):
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def jitter(self, x, sigma):
        # Jitter is added to every point in the time series data, so no change is needed based on channel_first
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    def __call__(self, x):
        return self.jitter(x, self.sigma)


class Scaling(object):
    def __init__(self, sigma=0.1, channel_first=False):
        self.sigma = sigma
        self.channel_first = channel_first

    def scaling(self, x, sigma):
        if self.channel_first:
            # For (B, C, L), generate a factor for each channel and broadcast across L
            factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[1], 1))
        else:
            # For (B, L, C), generate a factor for each channel and broadcast across L
            factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], 1, x.shape[2]))
        
        return np.multiply(x, factor)
    
    def __call__(self, x):
        return self.scaling(x, self.sigma)



