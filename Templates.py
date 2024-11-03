from abc import abstractmethod


class Model:
    def __init__(self):
        pass

    @abstractmethod
    def get_zcb_mse(self):
        raise NotImplementedError

    @abstractmethod
    def get_caplet_mse(self):
        raise NotImplementedError

    @abstractmethod
    def calibrate_model(self):
        raise NotImplementedError

    @abstractmethod
    def price_calplet(self):
        raise NotImplementedError

    @abstractmethod
    def price_cap(self):
        raise NotImplementedError

    @abstractmethod
    def price_swaption(self):
        raise NotImplementedError


class Curve:
    # For interest rate and volatility curves
    def __init__(self):
        pass

    @abstractmethod
    def interp(self):
        raise NotImplementedError


class Price_curve:
    def __init__(self):
        pass
