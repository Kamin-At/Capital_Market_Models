import numpy as np
from scipy import optimize
from scipy.stats import norm

EPSILON = 1e-7


def black_caplet_price(
    f,
    k,
    sigma,
    df,
    t,
    tau=0.25,
    N=1.0,
):
    """
    calculate caplet price with black's formula

    Args:
        f: (float) => forward rate
        k: (float) => strike rate
        sigma: (float) => Black volatility
        df: (float) => discount factor
        t: (float) => time to reset date in years
        tau: (float) => forward duration in years (default 0.25)
        N: (float) => notional amount (default 1.0)
    Returns:
        caplet price: (float)
    """
    d1 = (np.log(f / k) + (sigma**2) * t * 0.5) / ((sigma + EPSILON) * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return df * N * tau * (f * norm.cdf(d1) - k * norm.cdf(d2))


def get_black_caplet_iv(price, f, k, df, t, tau=0.25, N=1.0, initial_guess=0.3):
    """
    calculate caplet black's implied volatility

    Args:
        price: (float) => caplet price
        f: (float) => forward rate
        k: (float) => strike rate
        sigma: (float) => Black volatility
        df: (float) => discount factor
        t: (float) => time to reset date in years
        tau: (float) => forward duration in years (default 0.25)
        N: (float) => notional amount (default 1.0)
        initial_guess: (float) => initial guess of the Black implied volatility (default 0.3)
    Returns:
        iv: (float) => caplet implied volatility
    """
    iv = optimize.newton(
        lambda iv: price
        - black_caplet_price(f=f, k=k, sigma=iv, df=df, t=t, tau=tau, N=N),
        initial_guess,
    )

    return iv


def normal_caplet_price(
    f,
    k,
    sigma,
    df,
    t,
    tau=0.25,
    N=1.0,
):
    """
    calculate caplet price (from normal model)

    Args:
        f: (float) => forward rate
        k: (float) => strike rate
        sigma: (float) => Normal volatility
        df: (float) => discount factor
        t: (float) => time to reset date in years
        tau: (float) => forward duration in years (default 0.25)
        N: (float) => notional amount (default 1.0)
    Returns:
        caplet price: (float)
    """
    d = (f - k) / (sigma * np.sqrt(tau))

    return df * N * tau * ((f - k) * norm.cdf(d) + sigma * np.sqrt(tau) * norm.pdf(d))


def get_normal_caplet_iv(price, f, k, df, t, tau=0.25, N=1.0, initial_guess=0.01):
    """
    calculate caplet implied volatility (normal model)

    Args:
        price: (float) => caplet price
        f: (float) => forward rate
        k: (float) => strike rate
        sigma: (float) => Normal volatility
        df: (float) => discount factor
        t: (float) => time to reset date in years
        tau: (float) => forward duration in years (default 0.25)
        N: (float) => notional amount (default 1.0)
        initial_guess: (float) => initial guess of the Normal implied volatility (default 0.01)
    Returns:
        iv: (float) => Normal caplet implied volatility
    """
    iv = optimize.newton(
        lambda iv: price
        - normal_caplet_price(f=f, k=k, sigma=iv, df=df, t=t, tau=tau, N=N),
        initial_guess,
    )

    return iv


def black_cap_price(
    forward_curve,
    k,
    sigma,
    zcb_prices,
    time_to_reset_date,
    taus,
    N=1.0,
):
    """
    calculate cap price

    Args:
        forward_curve: (list[float]) => forward curve
        k: (float) => strike rate
        sigma: (float) => Black volatility
        zcb_prices: (list[float]) => discount factors
        time_to_reset_date: (list[float]) => time to reset date in years
        taus: (list[float]) => forward durations in years (default [0.25] * len(time_to_reset_date))
        N: (float) => notional amount (default 1.0)
    Returns:
        cap_price: (float) => cap price
    """

    assert (
        len(forward_curve) == len(zcb_prices)
        and len(zcb_prices) == len(time_to_reset_date)
        and len(time_to_reset_date) == len(taus)
    ), f"The legnths must be equal. len(forward_curve): {len(forward_curve)}, len(zcb_prices): {len(zcb_prices)}, len(time_to_reset_date): {len(zcb_prices)}, len(taus): {len(taus)}"

    cap_price = 0

    for i in range(len(zcb_prices)):
        caplet_price = black_caplet_price(
            forward_curve[i],
            k,
            sigma,
            zcb_prices[i],  # Discount at maturity not reset date
            time_to_reset_date[i],
            taus[i],
            N,
        )
        cap_price += caplet_price

    return cap_price


def get_black_cap_iv(
    price,
    forward_curve,
    k,
    zcb_prices,
    time_to_reset_date,
    taus=0.25,
    N=1.0,
    initial_guess=0.3,
):
    """
    calculate cap black's implied volatility

    Args:
        price: (float) => caplet price
        forward_curve: (list[float]) => forward curve
        k: (float) => strike rate
        zcb_prices: (list[float]) => discount factors
        time_to_reset_date: (list[float]) => time to reset date in years
        taus: (list[float]) => forward durations in years (default [0.25] * len(time_to_reset_date))
        N: (float) => notional amount (default 1.0)
        initial_guess: (float) => initial guess of the Black implied volatility (default 0.3)
    Returns:
        iv: (float) => cap implied volatility
    """

    sigma = optimize.newton(
        lambda iv: price
        - black_cap_price(
            forward_curve, k, iv, zcb_prices, time_to_reset_date, taus, N
        ),
        initial_guess,
    )

    return sigma


def spot_curve_to_zcb_curve(spot_curve, tenors):
    """
    convert spot_curve to zcb_curve

    Args:
        spot_curve: (list[float]) => spot curve
        tenors: (list[float]) => tenors
    Returns:
        zcb_curve: (list[float]) => zcb curves
    """

    assert len(spot_curve) == len(
        tenors
    ), f"len(spot_curve) [{len(spot_curve)}] != len(tenors) [{len(tenors)}]"

    zcb_curve = []
    for i in range(len(spot_curve)):
        zcb_curve.append(1.0 / (1.0 + spot_curve[i] * tenors[i]))

    return zcb_curve


def zcb_curve_to_spot_curve(zcb_curve, tenors):
    """
    convert zcb_curve to spot_curve

    Args:
        zcb_curve: (list[float]) => zcb curve
        tenors: (list[float]) => tenors
    Returns:
        spot_curve: (list[float]) => spot curves
    """
    assert len(zcb_curve) == len(
        tenors
    ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

    spot_curve = []
    for i in range(len(zcb_curve)):
        spot_curve.append((1.0 / zcb_curve[i] - 1.0) / tenors[i])

    return spot_curve


def zcb_curve_to_forward_curve(zcb_curve, tenors):
    """
    convert zcb_curve to forward_curve

    Args:
        zcb_curve: (list[float]) => zcb curve
        tenors: (list[float]) => tenors
    Returns:
        forward_curve: (list[float]) => forward curve
    """
    assert len(zcb_curve) == len(
        tenors
    ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

    zcb_curve = [1.0] + zcb_curve
    tenors = [0.0] + tenors

    forward_curve = []
    for i in range(len(zcb_curve) - 1):
        dt = tenors[i + 1] - tenors[i]
        forward_curve.append((zcb_curve[i] / zcb_curve[i + 1] - 1.0) / dt)
    forward_curve.append(np.nan)
    return forward_curve


def zcb_curve_to_forward_swap_curve(zcb_curve, tenors):
    """
    convert zcb_curve to forward_curve

    Args:
        zcb_curve: (list[float]) => zcb curve
        tenors: (list[float]) => tenors
    Returns:
        forward_swap_curve: (list[float]) => forward swap curve
    """
    assert len(zcb_curve) == len(
        tenors
    ), f"len(zcb_curve) [{len(zcb_curve)}] != len(tenors) [{len(tenors)}]"

    zcb_curve = [1.0] + zcb_curve
    tenors = [0.0] + tenors
    dt = [(tenors[i + 1] - tenors[i]) for i in range(len(tenors) - 1)]
    zcb_x_dt_cumsum_curve = np.cumsum(
        [dt[i] * zcb_curve[i + 1] for i in range(1, len(zcb_curve) - 1)]
    )
    delta_zcb = [zcb_curve[1] - zcb_curve[i] for i in range(2, len(zcb_curve))]

    assert len(zcb_x_dt_cumsum_curve) == len(
        delta_zcb
    ), f"len(zcb_x_dt_cumsum_curve) [{len(zcb_x_dt_cumsum_curve)}] != len(delta_zcb) [{len(delta_zcb)}]"

    forward_swap_curve = (
        [np.nan]
        + [delta_zcb[i] / zcb_x_dt_cumsum_curve[i] for i in range(len(delta_zcb))]
        + [np.nan]
    )

    return forward_swap_curve


if __name__ == "__main__":
    N = 1.0
    f = 0.041950
    k = 0.038863
    sigma = 0.008461329
    df = 0.977440241
    t = 0
    tau = 0.25

    result = normal_caplet_price(
        f,
        k,
        sigma,
        df,
        t,
        tau,
        N,
    )
    print()
