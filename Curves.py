from scipy.optimize import minimize
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline, interp1d
from Templates import *
from utils import *
from functools import cache


class Zcb_curve(Curve):
    def __init__(
        self,
        zcb_curve,
        tenors,
        interp_method,
    ):
        """
        Class constructor

        Args:
            zcb_curve: list[float] => zcb curves
            tenors: list[float] => tenors in years
            interp_method: str => "linear" or "cubic spline" (default 'linear')
        Returns:
            Zcb_curve object
        """
        Curve.__init__(self)
        assert len(zcb_curve) == len(tenors), "len(zcb_curve) must = len(tenors)"
        assert interp_method in [
            "cubic spline",
            "linear",
        ], "interp_method must be in ['cubic spline', 'linear']"

        self.zcb_curve = zcb_curve
        self.tenors = tenors
        self.forward_curve = zcb_curve_to_forward_curve(self.zcb_curve, self.tenors)
        self.forward_swap_curve = zcb_curve_to_forward_swap_curve(
            self.zcb_curve, self.tenors
        )

        self.interp_method = interp_method
        if self.interp_method == "cubic spline":
            self.interpolator = CubicSpline(
                [0.0] + self.tenors,
                [1.0] + self.zcb_curve,
                bounds_error=False,
                fill_value="extrapolate",
            )
        elif self.interp_method == "linear":
            self.interpolator = interp1d(
                [0.0] + self.tenors,
                [1.0] + self.zcb_curve,
                bounds_error=False,
                fill_value="extrapolate",
            )
        else:
            raise NotImplementedError(
                'interp_method must be "cubic spline" or "linear"'
            )

    def interp(self, T):
        """
        calculate the interpolation of the zcb value @ tenor T

        Args:
            T: float => tenor in years
        Returns:
            zcb: float => interpolated zcb price @ T
        """
        return self.interpolator(T)


class Vol_curve(Curve):
    def __init__(
        self,
        cap_price_curve,
        cap_black_vols,
        zcb_curve,
        tenors,
        interp_method,
        k=None,
        is_parametric=False,
        rho_function="simple exponential",
    ):
        """
        Class constructor

        Args:
            cap_price_curve: list[float] => ATM cap price curve
            cap_black_vols: list[float] => ATM cap vol curve
            zcb_curve: Zcb_curve => zcb curve object
            tenors: list[float] => tenors in years (reset date not the maturity) => actual/360
            interp_method: str => "piecewise constant" (default 'piecewise constant')
            k: float => strike rate
            is_parametric: Bool => If true, use parametric method
            rho_function: str => choose from ['simple exponential', 'complex exponential']
                                 simple exponential = exp(-beta * abs(t_i - t_j))
        Returns:
            Vol_curve object
        """
        Curve.__init__(self)
        assert (
            len(cap_price_curve) == len(tenors) - 1
        ), "len(cap_price_curve) must = len(tenors) - 1"
        assert len(zcb_curve) == len(tenors), "len(zcb_curve) must = len(tenors)"
        assert interp_method in [
            "piecewise constant",
        ], "interp_method must be in ['piecewise constant']"
        assert rho_function in [
            "simple exponential",
            "complex exponential",
        ], "rho_function must be in ['simple exponential', 'complex exponential']"

        self.cap_price_curve = cap_price_curve
        self.cap_black_vols = cap_black_vols
        self.zcb_curve = zcb_curve
        self.tenors = tenors
        self.interp_method = interp_method
        self.forward_swap_curve = zcb_curve_to_forward_swap_curve(
            self.zcb_curve, self.tenors
        )
        # If k is not given => assume that the given cap prices are ATM
        if type(k) == float:
            self.k = [k] * (len(self.forward_swap_curve) - 1)
        elif type(k) == list:
            self.k = np.array(k)
        elif k is None:
            self.k = self.forward_swap_curve[1:]
        else:
            raise NotImplementedError

        self.forward_curve = zcb_curve_to_forward_curve(self.zcb_curve, self.tenors)
        self.lmm_forward_swap_curve, self.lmm_w_matrix = LMM_zcb_curve_to_swap_curve(
            self.zcb_curve, self.tenors
        )
        self.interpolator = None
        self.dt = 0.001  # for interpolation grid
        self.is_parametric = is_parametric
        self.params = None
        self.rho_function = rho_function
        self.rho_params = None

    def interp(self, T, vol_type="black"):
        """
        calculate the interpolation of the zcb value @ tenor T
        Note: if the value out of range, use the last bondary value. (apply for both side)
        Args:
            T: float => tenor in years
            vol_type: str => choose from "black" (Black's model) and "normal" (Normal model)
        Returns:
            zcb: float => interpolated zcb price @ T
        """
        assert self.interp_method in [
            "piecewise constant"
        ], "Now only piecewise constant is available"
        if not self.interpolator:
            black_vol_grid = []
            normal_vol_grid = []
            tenor_grid = []
            ind = 0
            ind_dt = 0
            while ind < len(self.caplet_black_vols):
                tmp_t = ind_dt * self.dt
                if tmp_t >= self.tenors[ind]:
                    ind += 1
                    if ind >= len(self.caplet_black_vols):
                        break
                tenor_grid.append(tmp_t)
                if vol_type == "black":
                    black_vol_grid.append(self.caplet_black_vols[ind])
                elif vol_type == "normal":
                    normal_vol_grid.append(self.caplet_normal_vols[ind])
                else:
                    raise NotImplementedError("vol_type must be in ['black', 'normal']")
                ind_dt += 1
            if vol_type == "black":
                self.black_iv_interpolator = interp1d(
                    tenor_grid,
                    black_vol_grid,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=(self.caplet_black_vols[0], self.caplet_black_vols[-1]),
                )
            elif vol_type == "normal":
                self.normal_iv_interpolator = interp1d(
                    tenor_grid,
                    normal_vol_grid,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=(
                        self.caplet_normal_vols[0],
                        self.caplet_normal_vols[-1],
                    ),
                )
            else:
                raise NotImplementedError("vol_type must be in ['black', 'normal']")
        if vol_type == "black":
            return self.black_iv_interpolator(T)
        elif vol_type == "normal":
            return self.normal_iv_interpolator(T)
        else:
            raise NotImplementedError("vol_type must be in ['black', 'normal']")

    def generate_caplet_vol_term_structure_from_cap_vol(self):
        """
        Use cap prices to calculate caplet Black's and Normal vol term structures
        Idea:   tenor1 => sigma_cap_1 = sigma_caplet_1
                tenor2 => t2 * sigma_cap_2 ^ 2  = t1 * sigma_cap_1 ^ 2 + (t2 - t1) * sigma_caplet_2 ^ 2
                Solve for sigma_caplet_2
                Then keep going for all the cap vol => we then get the caplet vol term structure
                if is_parametric is True => optimize for its parameters after the piecewise constant.
        Args:
            -
        Returns:
            -
        """
        self.caplet_black_vols = []

        for ind in range(len(self.cap_black_vols)):
            tmp_cap_vol = self.cap_black_vols[ind]
            if ind == 0:
                self.caplet_black_vols.append(tmp_cap_vol)
            else:
                caplet_black_iv = (
                    (
                        self.tenors[ind] * (self.cap_black_vols[ind]) ** 2
                        - self.tenors[ind - 1] * (self.cap_black_vols[ind - 1]) ** 2
                    )
                    / (self.tenors[ind] - self.tenors[ind - 1])
                ) ** 0.5
                self.caplet_black_vols.append(caplet_black_iv)
            if self.is_parametric:
                self.params = self.solve_parametric()

    def generate_caplet_vol_term_structure(self):
        """
        Use cap prices to calculate caplet Black's and Normal vol term structures
        Idea:   tenor1 => Cap_1(sigma_cap_1) = Caplet_1(sigma_cap_1)
                tenor2 => Cap_2(sigma_cap_2) = Caplet_1(sigma_cap_2) + Caplet_2(sigma_cap_2) = Caplet_1(sigma_caplet_1) + Caplet_2(sigma_caplet_2)
                Solve for sigma_caplet_2
                tenor3 => Cap_3(sigma_cap_3) = Caplet_1(sigma_cap_3) + Caplet_2(sigma_cap_3) + Caplet_3(sigma_cap_3)
                                             = Caplet_1(sigma_caplet_1) + Caplet_2(sigma_caplet_2) + Caplet_3(sigma_caplet_3)
                Solve for sigma_caplet_3
                Then keep going for all the caplet prices => we then get the caplet vol term structure
                if is_parametric is True => optimize for its parameters after the piecewise constant.
        Args:
            -
        Returns:
            -
        """

        self.cap_black_vols = []
        self.caplet_black_vols = []
        self.caplet_normal_vols = []

        for ind in range(len(self.cap_price_curve)):
            tmp_cap_vol = get_black_cap_iv(
                self.cap_price_curve[ind],
                self.forward_curve[1 : ind + 2],
                self.k[ind],
                self.zcb_curve[1 : ind + 2],
                self.tenors[: ind + 1],
                taus=[0.25] * (ind + 1),
                N=1.0,
                initial_guess=0.3,
            )
            self.cap_black_vols.append(tmp_cap_vol)
            if ind == 0:
                self.caplet_black_vols.append(tmp_cap_vol)
                caplet_normal_iv = get_normal_caplet_iv(
                    self.cap_price_curve[ind],
                    self.forward_curve[ind + 1],
                    self.k[ind],
                    self.zcb_curve[ind + 1],
                    self.tenors[ind],
                    tau=0.25,
                    N=1.0,
                    initial_guess=0.01,
                )
                self.caplet_normal_vols.append(caplet_normal_iv)
            else:
                tmp_black_caplet_price = self.cap_price_curve[ind]
                for caplet_ind, caplet_vol in enumerate(self.caplet_black_vols):
                    tmp_caplet_price = black_caplet_price(
                        self.forward_curve[caplet_ind + 1],
                        self.k[ind],
                        caplet_vol,
                        self.zcb_curve[caplet_ind + 1],
                        self.tenors[caplet_ind],
                        tau=0.25,
                        N=1.0,
                    )
                    tmp_black_caplet_price -= tmp_caplet_price

                tmp_normal_caplet_price = self.cap_price_curve[ind]
                for caplet_ind, caplet_vol in enumerate(self.caplet_normal_vols):
                    tmp_caplet_price = normal_caplet_price(
                        self.forward_curve[caplet_ind + 1],
                        self.k[ind],
                        caplet_vol,
                        self.zcb_curve[caplet_ind + 1],
                        self.tenors[caplet_ind],
                        tau=0.25,
                        N=1.0,
                    )
                    tmp_normal_caplet_price -= tmp_caplet_price

                assert tmp_black_caplet_price > 0, "black_caplet_price must be positive"
                assert (
                    tmp_normal_caplet_price > 0
                ), "normal_caplet_price must be positive"

                caplet_black_iv = get_black_caplet_iv(
                    tmp_black_caplet_price,
                    self.forward_curve[caplet_ind + 2],
                    self.k[ind],
                    self.zcb_curve[caplet_ind + 2],
                    self.tenors[caplet_ind + 1],
                    tau=0.25,
                    N=1.0,
                    initial_guess=0.3,
                )
                caplet_normal_iv = get_normal_caplet_iv(
                    tmp_normal_caplet_price,
                    self.forward_curve[caplet_ind + 2],
                    self.k[ind],
                    self.zcb_curve[caplet_ind + 2],
                    self.tenors[caplet_ind + 1],
                    tau=0.25,
                    N=1.0,
                    initial_guess=0.01,
                )
                self.caplet_black_vols.append(caplet_black_iv)
                self.caplet_normal_vols.append(caplet_normal_iv)
        if self.is_parametric:
            self.params = self.solve_parametric()

    def parametric_function(self, params, T):
        """
        Compute the implied volatility function with parametric approach => sigma_t(T, a, b, c, d) = (a + b * T) * exp(-c * T) + d
        Args:
            params: list[double] (size of 4) => parameters a, b, c, d
            T: float => the reamaing time (from reset date)
        Returns:
            -
        """
        a, b, c, d = params
        return (a + b * T) * np.exp(-c * T) + d

    def solve_parametric(self):
        """
        Calibrate the parametric parameters a, b, c, d from the caplet volatility
        """

        def eval_parametric_params(params, tenors):
            error = 0.0
            prev_tenor = 0.0
            prev_cur_total_vol = 0.0
            prev_parametric_total_vol = 0.0
            for tenor in tenors:
                cur_total_vol = integrate.quad(
                    lambda x: self.interp(x) ** 2, prev_tenor, tenor
                )[0]
                parametric_total_vol = integrate.quad(
                    lambda x: self.parametric_function(params, x) ** 2,
                    prev_tenor,
                    tenor,
                )[0]
                prev_cur_total_vol += cur_total_vol
                prev_parametric_total_vol += parametric_total_vol
                error += (prev_cur_total_vol - prev_parametric_total_vol) ** 2
                prev_tenor = tenor

            print(error / len(tenors))
            return error / len(tenors)

        initial_params = np.array([0.1, 0.1, 0.1, 0.1])
        x = minimize(
            lambda params: eval_parametric_params(params, self.tenors),
            initial_params,
            method="Nelder-Mead",
            tol=1e-6,
        )
        print(x)
        return x

    def cal_rho(self, rho_params, t1, t2):
        """
        Compute the implied volatility function with parametric approach => rho_i_j = exp(-beta * |t1-t2|) for simple exponential
        Args:
            rho_params: list[double] (size of 1) => parameters beta
            t1: float => start contract time 1
            t2: float => start contract time 2
        Returns:
            rho: float => rho value
        """
        if self.rho_function == "simple exponential":
            beta = rho_params[0]
            return np.exp(-beta * np.abs(t1 - t2))
        elif self.rho_function == "complex exponential":
            rho, alpha, beta, kappa = rho_params
            rho_bar_min = rho * np.tanh(alpha * min(t1, t2))
            beta_min = beta * min(t1, t2) ** (-kappa)
            result = rho_bar_min + (1 - rho_bar_min) * np.exp(
                -beta_min * np.abs(t1 - t2)
            )
            return result
        else:
            raise NotImplementedError

    @cache
    def cal_integral_sigma1_sigma2(self, t, t1, t2):
        """
        Compute the integration from 0 to t of sigma_maturity_t1 * sigma_maturity_t2 dt
        Args:
            t: float => reset date
            t1: float => maturity t1
            t2: float => maturity t2
        Returns:
            result: float => the discrete integration result
        """
        result = 0.0
        n = int(t // self.tenors[0]) if t >= self.tenors[0] else 1
        prev_tenor = 0.0
        for ind in range(n):
            result += (
                (self.tenors[ind] - prev_tenor)
                * self.interp(np.abs(t2 - self.tenors[ind]))
                * self.interp(np.abs(t1 - self.tenors[ind]))
            )
            prev_tenor = self.tenors[ind]
        return result

    def cal_cov(self, rho_params, t1, t2):
        """
        Compute the integration from 0 to t of sigma_maturity_t1 * sigma_maturity_t2 dt
        Args:
            rho_params: array[float] => parameters for rho function
            t1: float => time from now to forward rate1
            t2: float => time from now to forward rate2
        Returns:
            result: float => the discrete integration result * rho()
        """
        return (
            self.cal_rho(rho_params, t1, t2)
            * self.interp(t1, vol_type="black")
            * self.interp(t2, vol_type="black")
        )

    def get_cov_matrix(self, rho_params, tenors):
        """
        Compute the covariance matrix of forward rates given their tenors
        Args:
            rho_params: array[float] => parameters for rho function
            tenors: list[float] => tenors of each forward rate
        Returns:
            cov_matrix: np.array[float, float] => covariance matrix of the forward rates
            chol: np.array[float, float] => cholesky decomposition of the cov_matrix
        """
        cov_matrix = np.zeros((len(tenors), len(tenors)))
        for i in range(len(tenors)):
            for j in range(len(tenors)):
                cov_matrix[i, j] = self.cal_cov(rho_params, tenors[i], tenors[j])
        chol = np.linalg.cholesky(cov_matrix)
        return cov_matrix, chol

    def cal_rebonato_formula(self, rho_params, t):
        """
        Compute rebonato_formula
        Args:
            rho_params: list[double] (size of 1) => parameters beta
            t: float => reset date
        Returns:
            result: float => the rebonato_formula result
        """

        contract_start_index = int(t // self.tenors[0])
        swap_rate = self.lmm_forward_swap_curve[contract_start_index]
        result = 0.0
        for ind1 in range(contract_start_index, len(self.tenors)):
            t1 = self.tenors[ind1]
            w1 = self.lmm_w_matrix[contract_start_index, ind1]
            for ind2 in range(contract_start_index, len(self.tenors)):
                t2 = self.tenors[ind2]
                w2 = self.lmm_w_matrix[contract_start_index, ind2]
                tmp_rho = self.cal_rho(rho_params, t1, t2)
                if (
                    np.abs(tmp_rho) > 1
                ):  # to make sure the output correlations are in [-1, 1]
                    return 1000000
                result += (
                    w1
                    * w2
                    * self.forward_curve[ind1]
                    * self.forward_curve[ind2]
                    * self.cal_integral_sigma1_sigma2(t, min(t1, t2), max(t1, t2))
                    * tmp_rho
                )
                print(
                    "t1:",
                    t1,
                    "w1:",
                    w1,
                    "t2:",
                    t2,
                    "w2:",
                    w2,
                    "f1:",
                    self.forward_curve[ind1],
                    "f2:",
                    self.forward_curve[ind2],
                    "integration val:",
                    self.cal_integral_sigma1_sigma2(t, min(t1, t2), max(t1, t2)),
                    "tmp_rho:",
                    tmp_rho,
                )
        return result / (swap_rate) ** 2

    def fit_corr_matrix(self, swaption_vol, swaption_reset_dates):
        """
        Calibrate the rho parametric function (parameter: beta) from the swaption volatility
        Args:
            swaption_vol: list[float] => swaption black volatilities
            swaption_reset_dates: list[float] => swaption reset dates
            maturity_date: float => the final maturity date (co-terminal concept => only 1 maturity date)
        Returns:
            fitted_rho_params: list[double] (size of 1) => parameters beta
        """
        assert len(swaption_vol) == len(
            swaption_reset_dates
        ), "len(swaption_vol) must be the same as len(swaption_reset_dates)."

        def eval_mse(rho_params):
            nonlocal swaption_vol, swaption_reset_dates
            mse = 0.0
            for ind in range(len(swaption_vol)):
                model_total_variance = self.cal_rebonato_formula(
                    rho_params, swaption_reset_dates[ind]
                )
                market_total_variance = (swaption_vol[ind] ** 2) * swaption_reset_dates[
                    ind
                ]
                print(
                    "market_total_variance:",
                    market_total_variance,
                    "model_total_variance:",
                    model_total_variance,
                )
                mse += (market_total_variance - model_total_variance) ** 2
            mse = mse / len(swaption_vol)
            print("mse:", mse)
            print("------------------------------------")
            return mse

        if self.rho_function == "simple exponential":
            rho_params = np.array([0.25])
        elif self.rho_function == "complex exponential":
            rho_params = np.array([0.8, 0.2, 0.2, 0.2])  # rho, alpha, beta, kappa
        else:
            raise NotImplementedError
        x = minimize(eval_mse, rho_params, method="Nelder-Mead", tol=1e-6)
        return x


if __name__ == "__main__":
    cap_price_curve = [
        0.000476999,
        0.001218952,
        0.001989746,
        0.002982203,
        0.004110025,
        0.00539507,
        0.006859078,
        0.008234175,
        0.009697875,
        0.011272431,
        0.012940934,
        0.014564239,
        0.016245701,
        0.017994826,
        0.019795692,
        0.021591312,
        0.02340779,
        0.025289734,
        0.027202241,
    ]

    tenors = [
        0.25,
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25,
        3.5,
        3.75,
        4,
        4.25,
        4.5,
        4.75,
        5.00,
    ]

    zcb_curve = [
        0.988412022,
        0.978829041,
        0.9708342,
        0.963157821,
        0.955975519,
        0.9489389,
        0.94188292,
        0.934884149,
        0.927855883,
        0.920949452,
        0.913943255,
        0.906990357,
        0.900043085,
        0.893140513,
        0.886216191,
        0.879345551,
        0.872459024,
        0.865692769,
        0.858830977,
        0.852023574,
    ]

    cap_vol = [
        0.2497,
        0.5639,
        0.4915,
        0.267480647,
        0.441841681,
        0.434286381,
        0.620374014,
        0.38351803,
        0.390164076,
        0.377284358,
        0.382985767,
        0.358733374,
        0.345141519,
        0.345690076,
        0.346994888,
        0.329358511,
        0.33403445,
        0.322512697,
        0.015554378,
    ]

    vol_curve = Vol_curve(
        cap_price_curve,
        cap_vol,
        zcb_curve,
        tenors,
        interp_method="piecewise constant",
        is_parametric=False,
        rho_function="simple exponential",
        # rho_function="complex exponential",
    )
    vol_curve.generate_caplet_vol_term_structure()
    # vol_curve.generate_caplet_vol_term_structure_from_cap_vol()# Not working
    b_vol = vol_curve.interp(0.1, vol_type="black")
    # swaption_vol = [0.31, 0.33, 0.35, 0.3, 0.25]
    # swaption_vol = [0.25, 0.3, 0.35, 0.33, 0.31]
    # swaption_vol = [0.26, 0.30, 0.35, 0.37, 0.30]
    # Actual swaption vol  @ 10/21/24
    # swaption_vol = [0.3693, 0.3147, 0.3072, 0.2998, 0.2963]
    # Actual swaption vol  @ 09/28/24
    swaption_vol = [0.3089, 0.3419, 0.338, 0.3282, 0.3269]
    swaption_reset_dates = [1 / 12, 1.0, 2.0, 3.0, 4.0]
    x = vol_curve.fit_corr_matrix(swaption_vol, swaption_reset_dates)
    cov_mat, chol = vol_curve.get_cov_matrix(x.x, [1, 2, 3, 4, 5])
    print("")
