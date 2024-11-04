from scipy.interpolate import CubicSpline, interp1d
from Templates import *
from utils import *


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
    def __init__(self, cap_price_curve, zcb_curve, tenors, interp_method, k=None):
        """
        Class constructor

        Args:
            cap_price_curve: list[float] => ATM cap price curves
            zcb_curve: Zcb_curve => zcb curve object
            tenors: list[float] => tenors in years (reset date not the maturity) => actual/360
            interp_method: str => "piecewise constant" (default 'piecewise constant')
            k: float => strike rate
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

        self.cap_price_curve = cap_price_curve
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
        self.interpolator = None
        self.dt = 0.001  # for interpolation grid

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
                black_vol_grid.append(self.caplet_black_vols[ind])
                normal_vol_grid.append(self.caplet_normal_vols[ind])
                ind_dt += 1

            self.black_iv_interpolator = interp1d(
                tenor_grid,
                black_vol_grid,
                kind="nearest",
                bounds_error=False,
                fill_value=(self.caplet_black_vols[0], self.caplet_black_vols[-1]),
            )
            self.normal_iv_interpolator = interp1d(
                tenor_grid,
                normal_vol_grid,
                kind="nearest",
                bounds_error=False,
                fill_value=(self.caplet_normal_vols[0], self.caplet_normal_vols[-1]),
            )
        if vol_type == "black":
            return self.black_iv_interpolator(T)
        elif vol_type == "normal":
            return self.normal_iv_interpolator(T)
        else:
            raise NotImplementedError("vol_type must be in ['black', 'normal']")

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
