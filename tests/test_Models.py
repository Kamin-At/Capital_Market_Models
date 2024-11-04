from Models import *
from Curves import *


def test_hw_caplet_price():
    PRECISION = 6

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

    zcb_curve = Zcb_curve(
        zcb_curve,
        tenors,
        interp_method="linear",
    )

    T = [
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
    ]

    S = [
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
    a = 1.21999850572609
    sigma = 0.0181861598520475
    k = 0.025
    P_0_T = [
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
    ]
    P_0_S = [
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

    n = 100
    hw_model = HW_model(zcb_curve, vol_curve, n=n)

    params = np.array([a, sigma])
    k = np.array(vol_curve.k)
    P_0_T = np.array(P_0_T)
    P_0_S = np.array(P_0_S)
    T = np.array(T)
    S = np.array(S)
    result = hw_model.hw_cap_price(params, k, P_0_T, P_0_S, T, S)
    # Cap vol
    answer = [
        0.002037697,
        0.004173838,
        0.006168939,
        0.007755576,
        0.009340397,
        0.010501228,
        0.011651041,
        0.012452666,
        0.01324417,
        0.014017372,
        0.014793643,
        0.015458021,
        0.016116346,
    ]
    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    for i in range(len(answer)):
        assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
            i
        ], f"value should be {answer} but got {result}"


def test_calibrate_a_and_sigma():
    PRECISION = 3

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

    a = 1.222
    sigma = 0.0181861598520475
    k = 0.025

    vol_curve = Vol_curve(
        cap_prices, zcb_curve, tenors, interp_method="piecewise constant", k=k
    )
    vol_curve.generate_caplet_vol_term_structure()

    zcb_curve = Zcb_curve(
        zcb_curve,
        tenors,
        interp_method="linear",
    )

    n = 100
    hw_model = HW_model(zcb_curve, vol_curve, n=n)

    k = np.array(hw_model.volatility_curve.k)
    P_0_T = np.array(hw_model.zcb_curve.zcb_curve[:-1])
    P_0_S = np.array(hw_model.zcb_curve.zcb_curve[1:])
    T = np.array(hw_model.zcb_curve.tenors[:-1])
    S = np.array(hw_model.zcb_curve.tenors[1:])

    hw_model.calibrate_a_and_sigma(k, P_0_T, P_0_S, T, S, is_without_constraint=True)

    answer = np.around(np.array([hw_model.a, hw_model.sigma]), PRECISION)
    result = np.around(np.array([a, sigma]), PRECISION)

    for i in range(len(answer)):
        assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
            i
        ], f"value should be {answer} but got {result}"


def test_calibrate_model():
    PRECISION = 6

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

    k = 0.025

    vol_curve = Vol_curve(
        cap_prices, zcb_curve, tenors, interp_method="piecewise constant", k=k
    )
    vol_curve.generate_caplet_vol_term_structure()

    zcb_curve = Zcb_curve(
        zcb_curve,
        tenors,
        interp_method="linear",
    )

    n = 100
    hw_model = HW_model(zcb_curve, vol_curve, n=n)
    hw_model.calibrate_model()

    answer = np.around(
        [zcb_curve.interp((i + 1) * hw_model.dt) for i in range(n)], PRECISION
    )
    result = np.around(hw_model.arrow_debreu_tree.sum(axis=1)[1:], PRECISION)

    for i in range(len(answer)):
        assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
            i
        ], f"value should be {answer} but got {result}"


def test_price_security():
    PRECISION = 6

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
    ]

    zcb_curve = [
        0.993719803,
        0.987330937,
        0.98083601,
        0.97433509,
        0.96802245,
        0.961750709,
        0.955519603,
        0.949328867,
        0.94317824,
        0.937067463,
        0.930996278,
        0.924964427,
    ]

    vol_curve = Vol_curve(
        cap_prices, zcb_curve, tenors, interp_method="piecewise constant", k=0.028
    )
    vol_curve.generate_caplet_vol_term_structure()

    zcb_curve = Zcb_curve(
        zcb_curve,
        tenors,
        interp_method="linear",
    )
    n = 36

    hw_model = HW_model(zcb_curve, vol_curve, n)
    # hw_model.calibrate_model()
    k = np.array(hw_model.volatility_curve.k)
    P_0_T = np.array(hw_model.zcb_curve.zcb_curve[:-1])
    P_0_S = np.array(hw_model.zcb_curve.zcb_curve[1:])
    T = np.array(hw_model.zcb_curve.tenors[:-1])
    S = np.array(hw_model.zcb_curve.tenors[1:])

    hw_model.calibrate_a_and_sigma(k, P_0_T, P_0_S, T, S)
    hw_model.a = 0.5
    hw_model.sigma = 0.012207

    hw_model.calibrate_trinomial_tree()

    caplet = Caplet(
        notional=1.0,
        reset_time=2,
        maturity_time=3,
        strike=0.04,
        tau=0.25,
        is_call=False,
    )
    price = hw_model.price_security(caplet, for_test=True)[1]

    answer = 0.000171285545718511

    answer = np.around(answer, PRECISION)
    result = np.around(price, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"
