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
            n: int => number of time steps
        """
        self.zcb_curve = zcb_curve
        self.volatility_curve = volatility_curve
        self.n = n  # Max tenor
        self.dt = self.zcb_curve.tenors[-1] / self.n

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

    def calibrate_model(self):
        # All the values observed from the market
        k = np.array(self.volatility_curve.k)
        P_0_T = np.array(self.zcb_curve.zcb_curve[:-1])
        P_0_S = np.array(self.zcb_curve.zcb_curve[1:])
        T = np.array(self.zcb_curve.tenors[:-1])
        S = np.array(self.zcb_curve.tenors[1:])

        self.calibrate_a_and_sigma(k, P_0_T, P_0_S, T, S)
        self.calibrate_trinomial_tree()

    def calibrate_a_and_sigma(self, k, P_0_T, P_0_S, T, S, is_without_constraint=False):
        """
        calibrate the mean reversion a and the short rate normal volatility sigma to the market cap prices
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
            -
        """
        initial_guess = np.array([0.02, 0.01])  # a and sigma
        if is_without_constraint:
            x = minimize(
                lambda x: (self.get_mpse_cap(x, k, P_0_T, P_0_S, T, S)),
                initial_guess,
                method="Nelder-Mead",
            )
        else:
            x = minimize(
                lambda x: (
                    100000
                    if x[0] <= 0.00 or x[1] <= 0.005 or x[0] >= 0.5 or x[1] >= 0.03
                    else self.get_mpse_cap(x, k, P_0_T, P_0_S, T, S)
                ),
                initial_guess,
                method="Nelder-Mead",
            )
        print("calibration results:")
        print(x)
        print("Model cap prices:", self.hw_cap_price(x.x, k, P_0_T, P_0_S, T, S))
        print("Actual cap prices:", self.volatility_curve.cap_price_curve)
        print(
            "Mean Percentage Square Error [in %]:",
            self.get_mpse_cap(x.x, k, P_0_T, P_0_S, T, S) * 100,
        )
        self.a = x.x[0]
        self.sigma = x.x[1]
        self.dr_star = np.sqrt(3 * self.sigma**2 * self.dt)

    def calculate_transition_prob(self, j):
        if self.j_min < j < self.j_max:
            p_up1 = 1 / 6 + (j**2 * self.M**2 + j * self.M) / 2
            p_up2 = 0.0
            p_mid = 2 / 3 - j**2 * self.M**2
            p_down1 = 1 / 6 + (j**2 * self.M**2 - j * self.M) / 2
            p_down2 = 0.0
        elif j <= self.j_min:
            p_up1 = -1 / 3 - j**2 * self.M**2 + 2 * j * self.M
            p_up2 = 1 / 6 + (j**2 * self.M**2 - j * self.M) / 2
            p_mid = 7 / 6 + (j**2 * self.M**2 - 3 * j * self.M) / 2
            p_down1 = 0.0
            p_down2 = 0.0
        else:
            p_down1 = -1 / 3 - j**2 * self.M**2 - 2 * j * self.M
            p_down2 = 1 / 6 + (j**2 * self.M**2 + j * self.M) / 2
            p_mid = 7 / 6 + (j**2 * self.M**2 + 3 * j * self.M) / 2.0
            p_up1 = 0.0
            p_up2 = 0.0
        return p_down2, p_down1, p_mid, p_up1, p_up2

    def initilize_transition_matrix(self):
        self.transition_prob_matrix = np.zeros((2 * self.n + 1, 5))
        self.M = np.exp(-self.a * self.dt) - 1
        self.V = (self.sigma**2) * (1 - np.exp(-2 * self.a * self.dt)) / (2 * self.a)
        for i in range(2 * self.n + 1):
            for ind, prob_j in enumerate(self.calculate_transition_prob(i - self.n)):
                self.transition_prob_matrix[i][ind] = prob_j

    def calibrate_trinomial_tree(self):
        self.rate_tree = np.zeros((self.n, 2 * self.n + 1))
        self.arrow_debreu_tree = np.zeros((self.n, 2 * self.n + 1))
        self.arrow_debreu_tree[0, self.n] = 1.0
        self.alpha = np.zeros(self.n)
        self.alpha[0] = -np.log(self.zcb_curve.interp(self.dt)) / self.dt
        self.rate_tree[0, self.n] = self.alpha[0]
        self.j_max = int(0.184 / (self.a * self.dt))
        self.j_min = -int(0.184 / (self.a * self.dt))

        self.initilize_transition_matrix()
        self.j_vector = np.arange(2 * self.n + 1) - self.n
        self.dr_dt_discount = np.exp(-(self.j_vector) * self.dr_star * self.dt)

        for t in range(1, self.n):
            for j in range(self.n - t, self.n + t + 1):
                if j - 2 >= 0:  # from j - 2 => 2 up
                    self.arrow_debreu_tree[t, j] += (
                        self.arrow_debreu_tree[t - 1, j - 2]
                        * self.transition_prob_matrix[j - 2, 4]
                        / np.exp(self.rate_tree[t - 1, j - 2] * self.dt)
                    )
                if j - 1 >= 0:  # from j - 1 => 1 up
                    self.arrow_debreu_tree[t, j] += (
                        self.arrow_debreu_tree[t - 1, j - 1]
                        * self.transition_prob_matrix[j - 1, 3]
                        / np.exp(self.rate_tree[t - 1, j - 1] * self.dt)
                    )
                self.arrow_debreu_tree[t, j] += (
                    self.arrow_debreu_tree[t - 1, j]
                    * self.transition_prob_matrix[j, 2]
                    / np.exp(self.rate_tree[t - 1, j] * self.dt)
                )  # from j => mid
                if j + 1 < self.arrow_debreu_tree.shape[1]:
                    self.arrow_debreu_tree[t, j] += (
                        self.arrow_debreu_tree[t - 1, j + 1]
                        * self.transition_prob_matrix[j + 1, 1]
                        / np.exp(self.rate_tree[t - 1, j + 1] * self.dt)
                    )  # from j + 1 => 1 down
                if j + 2 < self.arrow_debreu_tree.shape[1]:
                    self.arrow_debreu_tree[t, j] += (
                        self.arrow_debreu_tree[t - 1, j + 2]
                        * self.transition_prob_matrix[j + 2, 0]
                        / np.exp(self.rate_tree[t - 1, j + 2] * self.dt)
                    )  # from j + 2 => 2 down
            self.alpha[t] = (
                np.log(
                    (self.arrow_debreu_tree[t] * self.dr_dt_discount).sum()
                    / self.zcb_curve.interp(self.dt * (t + 1))
                )
                / self.dt
            )
            self.rate_tree[t] = self.alpha[t] + self.dr_star * self.j_vector

    def eval_a_and_sigma(self, a, sigma, k, P_0_T, P_0_S, T, S, return_prices=False):
        """
        calculate calculate mean square error for calibration
        Args:
            a: float => mean reversion speed parameter
            sigma: float => volatility parameter
            return_prices: bool => return MSE with prices (if True => return (MSE, prices) else => return MSE)
        Returns:
            MSE: mean square error of the calibration
            prices: model prices
        """
        model_prices = self.hw_caplet_price(a, sigma, k, P_0_T, P_0_S, T, S)
        return model_prices - self.volatility_curve


if __name__ == "__main__":
    # Cap prices
    cap_prices = [
        0.001928473,
        0.003954714,
        0.005964848,
        0.008155229,
        0.009895812,
        0.011157567,
        0.01237674,
        0.012956468,
        0.013612125,
        0.013929541,
        0.014375395,
        0.014894215,
        0.015374702,
    ]

    tenors = [
        0.261111111111,
        0.513888888889,
        0.758333333333,
        1.019444444444,
        1.272222222222,
        1.525000000000,
        1.777777777778,
        2.030555555556,
        2.286111111111,
        2.541666666667,
        2.791666666667,
        3.044444444444,
        3.300000000000,
        3.555555555556,
    ]

    zcb_curve = [
        0.993294243,
        0.985145961,
        0.977224557,
        0.969227104,
        0.962049506,
        0.954935844,
        0.94855094,
        0.942223867,
        0.936570281,
        0.93096599,
        0.925516089,
        0.920038016,
        0.914804077,
        0.909605921,
    ]

    vol_curve = Vol_curve(
        cap_prices, zcb_curve, tenors, interp_method="piecewise constant", k=0.025
    )
    vol_curve.generate_caplet_vol_term_structure()

    Zcb_curve = Zcb_curve(
        zcb_curve,
        tenors,
        interp_method="linear",
    )
    n = 100

    hw_model = HW_model(Zcb_curve, vol_curve, n)
    hw_model.calibrate_model()
    print()
