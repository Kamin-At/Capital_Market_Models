from Templates import *
from Securities import *
from utils import *
from scipy.stats import norm
from Curves import *
from scipy.optimize import minimize
from functools import cache
import sys

sys.setrecursionlimit(2000)


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
        self.rate_tree = None
        self.alpha = None
        self.arrow_debreu_tree = None

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
        """
        1.calibrate a and sigma to the cap market prices
        2.calibrate alpha(t) with trinomial tree algorithm
        Args:
            -
        Returns:
            -
        """
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
        """
        Calculate the transition probability for level j moving from timestep t to t+1
        Args:
            j: int => the level of the current node
        Returns:
            p_down2: float => probability of going down by 2 levels in the next timestep
            p_down1: float => probability of going down by 1 level in the next timestep
            p_mid: float => probability of no moving in the next timestep
            p_up1: float => probability of going up by 1 levels in the next timestep
            p_up2: float => probability of going up by 2 levels in the next timestep
        """
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
        """
        Calculate the transition probability matrix for all the levels moving from timestep t to t+1.
        This is for memoization.
        Args:
            -
        Returns:
            -
        """
        self.transition_prob_matrix = np.zeros((2 * self.n + 1, 5))
        self.M = np.exp(-self.a * self.dt) - 1
        self.V = (self.sigma**2) * (1 - np.exp(-2 * self.a * self.dt)) / (2 * self.a)
        self.dr_star = np.sqrt(3 * self.V)
        for i in range(2 * self.n + 1):
            for ind, prob_j in enumerate(self.calculate_transition_prob(i - self.n)):
                self.transition_prob_matrix[i][ind] = prob_j

    def calibrate_trinomial_tree(self):
        """
        Calculate trinomial trees (arrow_debreu_tree, and rate_tree) and alpha(t).
        Args:
            -
        Returns:
            -
        """
        self.rate_tree = np.zeros((self.n + 1, 2 * self.n + 1))
        self.arrow_debreu_tree = np.zeros((self.n + 1, 2 * self.n + 1))
        self.arrow_debreu_tree[0, self.n] = 1.0
        self.alpha = np.zeros(self.n + 1)
        self.alpha[0] = -np.log(self.zcb_curve.interp(self.dt)) / self.dt
        self.rate_tree[0, self.n] = self.alpha[0]
        self.j_max = -0.184 / (np.exp(-self.a * self.dt) - 1)
        if self.j_max != float(int(self.j_max)):
            self.j_max = int(self.j_max) + 1
        self.j_min = -self.j_max

        self.initilize_transition_matrix()
        self.j_vector = np.arange(2 * self.n + 1) - self.n
        self.dr_dt_discount = np.exp(-(self.j_vector) * self.dr_star * self.dt)

        # Moving from timestep 1 to self.n to find all the alpha(t) that reproduce the given zcb_curve
        for t in range(1, self.n + 1):
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

    def trinomial_algorithm(
        self,
        notional,
        cashflows,
        exercisable_times,
        is_call,
        for_test=False,
    ):
        """
        Trinomial algorithm to perform security pricing with recursion and memoization techniques
        Args:
            notional: float => notional amount
            cashflows: np.array[float] => cashflows at each timestep. Ex: coupon or premium payments
            exercisable_times: np.array[float] => exercise flag for each time step (0 = not exercisable)
            is_call: bool => is a call option (for swaption, True = pay fixed, False = pay floating)
            for_test: bool => flag for unit test (very specific with true price => Reference: Prof. Neil D. Pearson)
        Returns:
            bond_value: float => fair bond price (used during price calculations)
            option_value: float => security value (for now only European, American and Bermudan swaptions and
                          simple caplets (for simeple test case) are available)
        """

        @cache
        def trinomial_pricing(i, j):
            nonlocal notional, cashflows, exercisable_times, is_call
            if i == self.n:
                return cashflows[self.n], 0.0
            bond_value = 0.0
            option_value = 0.0
            # going up 2 levels
            if j + 2 <= self.n + (i + 1):
                tmp_bond_value, tmp_option_value = trinomial_pricing(i + 1, j + 2)
                bond_value += tmp_bond_value * self.transition_prob_matrix[j, 4]
                option_value += tmp_option_value * self.transition_prob_matrix[j, 4]
            # going up 1 level
            if j + 1 <= self.n + (i + 1):
                tmp_bond_value, tmp_option_value = trinomial_pricing(i + 1, j + 1)
                bond_value += tmp_bond_value * self.transition_prob_matrix[j, 3]
                option_value += tmp_option_value * self.transition_prob_matrix[j, 3]
            # idle
            tmp_bond_value, tmp_option_value = trinomial_pricing(i + 1, j)
            bond_value += tmp_bond_value * self.transition_prob_matrix[j, 2]
            option_value += tmp_option_value * self.transition_prob_matrix[j, 2]
            # going down 1 level
            if j - 1 >= self.n - (i + 1):
                tmp_bond_value, tmp_option_value = trinomial_pricing(i + 1, j - 1)
                bond_value += tmp_bond_value * self.transition_prob_matrix[j, 1]
                option_value += tmp_option_value * self.transition_prob_matrix[j, 1]
            # going down 2 levels
            if j - 2 >= self.n - (i + 1):
                tmp_bond_value, tmp_option_value = trinomial_pricing(i + 1, j - 2)
                bond_value += tmp_bond_value * self.transition_prob_matrix[j, 0]
                option_value += tmp_option_value * self.transition_prob_matrix[j, 0]

            bond_value *= np.exp(-self.rate_tree[i, j] * self.dt)
            bond_value += cashflows[i]
            option_value *= np.exp(-self.rate_tree[i, j] * self.dt)

            if for_test:
                if is_call:
                    value_if_exercise = notional * max(bond_value - 0.96, 0.0)
                else:
                    value_if_exercise = notional * max(0.96 - bond_value, 0.0)
            else:
                value_if_exercise = (
                    max(notional - bond_value, 0.0)
                    if is_call
                    else max(bond_value - notional, 0.0)
                )
            if exercisable_times[i]:
                option_value = max(option_value, value_if_exercise)
            return bond_value, option_value

        return trinomial_pricing(0, self.n)

    def price_security(self, Security, for_test=False):
        """
        Calculate the Security price.
        Args:
            Security: Security => object from Security class in template
        Returns:
            price: float => the spwation price from HW model
        """
        assert (
            type(self.rate_tree) is not None
        ), "Calibration must be done before pricing"
        assert self.alpha is not None, "Calibration must be done before pricing"
        assert (
            self.arrow_debreu_tree is not None
        ), "Calibration must be done before pricing"
        cashflows = Security.get_cashflows(self.dt, self.n)
        exercisable_times = Security.generate_exercisable_times(self.dt, self.n)
        # swaption_price = self.trinomial_algorithm(
        #     swaption.notional, cashflows, exercisable_times, swaption.is_pay_fixed
        # )
        # return swaption_price
        price = self.trinomial_algorithm(
            Security.notional,
            cashflows,
            exercisable_times,
            Security.is_call,
            for_test=for_test,
        )
        return price


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
    n = 200
    reset_time = 1.0
    hw_model = HW_model(Zcb_curve, vol_curve, n)
    hw_model.calibrate_model()
    swaption = Swaption(
        notional=1.0,
        reset_time=reset_time,
        maturity_time=3,
        fixed_premium_rate=0.025,
        premium_frequency=4,
        coupon_times=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00],
        is_pay_fixed=True,
        exercisable_times=[reset_time],
        exercisable_after_time=None,
    )

    price = hw_model.price_security(swaption)
    print(price)
    print()
