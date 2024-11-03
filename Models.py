from Templates import *
from utils import *
from scipy.stats import norm
from Curves import *
from scipy.optimize import minimize
from functools import lru_cache


class HW_model(Model):
    """
    Hull White 1-factor model implementation
    """

    def __init__(self, zcb_curve, volatility_curve, n):
        """
        Hull White 1-factor model implementation
        Args:
            zcb_curve: Zcb_curve => the Zcb_curve class object
            volatility_curve: Vol_curve => the Vol_curve class object
            n: int => number of timesteps
        """
        self.zcb_curve = zcb_curve
        self.volatility_curve = volatility_curve
        self.n = n
        self.dt = self.zcb_curve.tenors[-1] / self.n  # Max tenor

    def hw_caplet_price(self, a, sigma, k, P_0_T, P_0_S, T, S):
        """
        calculate the cap price with analytical HW solution

        Args:
            a: float => mean reversion speed parameter
            sigma: float => volatility parameter
            k: float => strike rate
            P_0_T: float => zcb price from t = 0 to reset date
            P_0_S: float => zcb price from t = 0 to maturity date
            T: float => time to reset date
            S: float => time to maturity date
        Returns:
            price: float => cap price from the analytical solution
        """
        X = 1 + k * (S - T)
        B = (1 - np.exp(-a * (S - T))) / a
        sigma_p = sigma * B * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a))
        d1 = np.log(X * P_0_S / (P_0_T)) / sigma_p + sigma_p * 0.5
        d2 = d1 - sigma_p
        return P_0_T * norm.cdf(-d2) - X * P_0_S * norm.cdf(-d1)

    def hw_cap_price(self, params, k, P_0_T, P_0_S, T, S):
        """
        calculate the cap price with analytical HW solution
        Args:
            params: tuple[float, float]
                a: float => mean reversion speed parameter
                sigma: float => volatility parameter
            k: float => strike rate
            P_0_T: float => zcb price from t = 0 to reset date
            P_0_S: float => zcb price from t = 0 to maturity date
            T: float => time to reset date
            S: float => time to maturity date
        Returns:
            price: float => cap price from the analytical solution
        """
        a, sigma = params
        cap_prices = []
        for i in range(1, len(P_0_T) + 1):
            tmp_cap_price = self.hw_caplet_price(
                a, sigma, k[i - 1], P_0_T[:i], P_0_S[:i], T[:i], S[:i]
            )
            cap_prices.append(tmp_cap_price.sum())
        return np.array(cap_prices)

    def get_mpse_cap(self, params, k, P_0_T, P_0_S, T, S):
        """
        calculate mean percentage square error between the HW cap prices and the market cap prices
        Args:
            params: tuple[float, float]
                a: float => mean reversion speed parameter
                sigma: float => volatility parameter
            k: float => strike rate
            P_0_T: float => zcb price from t = 0 to reset date
            P_0_S: float => zcb price from t = 0 to maturity date
            T: float => time to reset date
            S: float => time to maturity date
        Returns:
            mpse: float => mean percentage square error
        """
        return np.mean(
            (
                (
                    self.hw_cap_price(params, k, P_0_T, P_0_S, T, S)
                    - self.volatility_curve.cap_price_curve
                )
                / self.volatility_curve.cap_price_curve
            )
            ** 2
        )
