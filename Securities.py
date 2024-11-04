from Templates import *
import numpy as np


class Swaption(Security):
    def __init__(
        self,
        notional,
        reset_time,
        maturity_time,
        fixed_premium_rate,
        premium_frequency,
        is_pay_fixed,
        exercisable_times=None,
        exercisable_after_time=None,
    ):
        Security.__init__(self)
        self.notional = notional
        self.reset_time = reset_time
        self.maturity_time = maturity_time
        self.fixed_premium_rate = fixed_premium_rate
        self.premium_frequency = premium_frequency
        self.exercisable_times = exercisable_times
        self.exercisable_after_time = exercisable_after_time
        self.is_pay_fixed = is_pay_fixed

    def generate_exercisable_times(self, dt, n):
        """
        create is_exercisable array where 1 mean the swaption is exercisable at the time step.
        Args:
            dt: float => time stepsize
            n: int => number of timesteps
        Returns:
            is_exercisable: np.array[np.float] => 1 = exercisable else no exercisable
        """
        is_exercisable = np.arange(n) * dt
        if self.exercisable_after_time:
            condition = (is_exercisable >= self.exercisable_after_time) & (
                is_exercisable <= self.reset_time
            )
            is_exercisable = np.where(condition, 1.0, 0.0)
        else:
            is_exercisable = np.zeros(n)
        if self.exercisable_times:
            for i in self.exercisable_times:
                is_exercisable[int(i // dt)] = 1.0
        return is_exercisable

    def get_cashflows(self, dt, n):
        """
        create cashflows array indicating cashflows at each timestep.
        Args:
            dt: float => time stepsize
            n: int => number of timesteps
        Returns:
            cashflows: np.array[np.float] => cashflows at each timestep
        """
        tmp_time = self.reset_time + 1 / self.premium_frequency
        cashflows = np.zeros(n)
        while tmp_time <= self.maturity_time:
            cashflows[int(tmp_time // dt)] = (
                self.notional * self.fixed_premium_rate / self.premium_frequency
            )
            tmp_time += 1 / self.premium_frequency
        cashflows[int((tmp_time - 1 / self.premium_frequency) // dt)] += self.notional
        return cashflows
