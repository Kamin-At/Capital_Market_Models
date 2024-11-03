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
            self.interpolator = CubicSpline(self.tenors, self.zcb_curve)
        elif self.interp_method == "linear":
            self.interpolator = interp1d(self.tenors, self.zcb_curve)
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
