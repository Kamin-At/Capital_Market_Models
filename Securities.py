from Templates import *


class Swaption:
    def __init__(
        self,
        reset_time,
        maturity_time,
        fixed_premium_rate,
        premium_frequency,
        is_bermudan,
    ):
        self.reset_time = reset_time
        self.maturity_time = maturity_time
        self.fixed_premium_rate = fixed_premium_rate
        self.premium_frequency = premium_frequency
        self.is_bermudan = is_bermudan
