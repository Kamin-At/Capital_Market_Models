from utils import *
import numpy as np


def test_black_caplet_price():
    PRECISION = 7

    tau = 0.25
    df = 0.955975519
    f = 0.0300522
    k = 0.03353653
    sigma = 0.301687537
    t = 1.0000
    N = 1.0

    result = black_caplet_price(f, k, sigma, df, t, tau, N)
    answer = 0.000553777

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"


def test_get_black_iv():
    PRECISION = 5

    price = 0.000553777
    tau = 0.25
    df = 0.955975519
    f = 0.0300522
    k = 0.03353653
    t = 1.0000
    N = 1.0

    result = get_black_caplet_iv(price, f, k, df, t, tau, N)
    answer = 0.301687537

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"


def test_normal_caplet_price():
    PRECISION = 5

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
    answer = 0.0008947126951

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"


def test_get_normal_iv():
    PRECISION = 5

    price = 0.0008947126951
    N = 1.0
    f = 0.041950
    k = 0.038863
    df = 0.977440241
    t = 0
    tau = 0.25

    result = get_normal_caplet_iv(price, f, k, df, t, tau, N)
    answer = 0.008461329

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"


def test_black_cap_price():
    PRECISION = 7

    forward_curve = [
        0.039161,
        0.032940089,
        0.031880044,
        0.030052244,
        0.029661,
        0.029965422,
        0.029944978,
        0.030298956,
        0.029997,
        0.0306636,
        0.0306636,
        0.030875289,
        0.030913711,
        0.031253422,
        0.031253422,
        0.031572956,
        0.031264,
        0.031958756,
        0.031958756,
    ]
    zcb_prices = [
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
    time_to_reset_date = [
        0.2500,
        0.5000,
        0.7500,
        1.0000,
        1.2500,
        1.5000,
        1.7500,
        2.0000,
        2.2500,
        2.5000,
        2.7500,
        3.0000,
        3.2500,
        3.5000,
        3.7500,
        4.0000,
        4.2500,
        4.5000,
        4.7500,
    ]

    sigma = 0.3364
    taus = [0.25] * len(time_to_reset_date)
    k = 0.0314042141880721
    N = 1.0

    result = black_cap_price(
        forward_curve, k, sigma, zcb_prices, time_to_reset_date, taus, N
    )
    answer = 0.027202241

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"


def test_get_black_cap_iv():
    PRECISION = 4

    forward_curve = [
        0.039161,
        0.032940089,
        0.031880044,
        0.030052244,
        0.029661,
        0.029965422,
        0.029944978,
        0.030298956,
        0.029997,
        0.0306636,
        0.0306636,
        0.030875289,
        0.030913711,
        0.031253422,
        0.031253422,
        0.031572956,
        0.031264,
        0.031958756,
        0.031958756,
    ]
    zcb_prices = [
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
    time_to_reset_date = [
        0.2500,
        0.5000,
        0.7500,
        1.0000,
        1.2500,
        1.5000,
        1.7500,
        2.0000,
        2.2500,
        2.5000,
        2.7500,
        3.0000,
        3.2500,
        3.5000,
        3.7500,
        4.0000,
        4.2500,
        4.5000,
        4.7500,
    ]

    price = 0.027202241
    taus = [0.25] * len(time_to_reset_date)
    k = 0.0314042141880721
    N = 1.0

    result = get_black_cap_iv(
        price, forward_curve, k, zcb_prices, time_to_reset_date, taus, N
    )
    answer = 0.3364

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    assert result == answer, f"value should be {answer} but got {result}"


def test_spot_curve_to_zcb_curve():
    PRECISION = 7

    spot_curve = [0.0534157, 0.0510155, 0.0478062, 0.0428661]
    tenors = [0.0861111111, 0.2555555556, 0.5111111111, 1.0138888889]
    result = spot_curve_to_zcb_curve(spot_curve, tenors)
    answer = [0.995421375, 0.987130489, 0.976148514, 0.958348761]

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    for i in range(len(answer)):
        assert result[i] == answer[i], f"value should be {answer} but got {result}"


def test_zcb_curve_to_spot_curve():
    PRECISION = 7

    zcb_curve = [0.995421375, 0.987130489, 0.976148514, 0.958348761]
    tenors = [0.0861111111, 0.2555555556, 0.5111111111, 1.0138888889]
    result = zcb_curve_to_spot_curve(zcb_curve, tenors)
    answer = [0.0534157, 0.0510155, 0.0478062, 0.0428661]

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    for i in range(len(answer)):
        assert result[i] == answer[i], f"value should be {answer} but got {result}"


def test_zcb_curve_to_forward_curve():
    PRECISION = 7

    zcb_curve = [0.995421375, 0.987130489, 0.976148514, 0.958348761]
    tenors = [0.0861111111, 0.2555555556, 0.5111111111, 1.0138888889]
    result = zcb_curve_to_forward_curve(zcb_curve, tenors)

    answer = [0.0534157, 0.0495677, 0.0440230, 0.0369415, np.nan]

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    for i in range(len(answer)):
        assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
            i
        ], f"value should be {answer} but got {result}"


def test_zcb_curve_to_forward_swap_curve():
    PRECISION = 6

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
    tenors = [
        0.2500,
        0.5000,
        0.7500,
        1.0000,
        1.2500,
        1.5000,
        1.7500,
        2.0000,
        2.2500,
        2.5000,
        2.7500,
        3.0000,
        3.2500,
        3.5000,
        3.7500,
        4.0000,
        4.2500,
        4.5000,
        4.7500,
        5.0000,
    ]
    result = zcb_curve_to_forward_swap_curve(zcb_curve, tenors)

    answer = [
        np.nan,
        0.039161,
        0.036063299,
        0.034680058,
        0.03353653,
        0.032773175,
        0.032314017,
        0.031983182,
        0.031778164,
        0.031586159,
        0.031497003,
        0.031424071,
        0.031380222,
        0.03134595,
        0.031339663,
        0.031334216,
        0.031348294,
        0.031343635,
        0.031375614,
        0.031404214,
        np.nan,
    ]

    answer = np.around(answer, PRECISION)
    result = np.around(result, PRECISION)

    for i in range(len(answer)):
        assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
            i
        ], f"value should be {answer} but got {result}"


def test_LMM_zcb_curve_to_swap_curve():
    PRECISION = 6

    zcb_curve = [
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
    tenors = [
        0.2500,
        0.5000,
        0.7500,
        1.0000,
        1.2500,
        1.5000,
        1.7500,
        2.0000,
        2.2500,
        2.5000,
        2.7500,
        3.0000,
        3.2500,
        3.5000,
        3.7500,
        4.0000,
        4.2500,
        4.5000,
        4.7500,
    ]
    result1, result2 = LMM_zcb_curve_to_swap_curve(zcb_curve, tenors)
    # Ex. if contract_start_time = 0.25 => the first cashflow of the swap(contract_start_time, final_cashflow_time) is at 0.5
    # final_cashflow_time is always set to the last value in the provided tenor
    answer1 = (
        [  # (contract_start_time, index, S(contract_start_time, final_cashflow_time))
            (
                0.25,
                0,
                0.031059241,
            ),
            (1.00, 3, 0.031045668),
            (2.00, 7, 0.031232946),
        ]
    )
    # Test for forward swap rate
    for _, ind, tmp_answer in answer1:
        assert (np.isnan(result1[ind]) and np.isnan(tmp_answer)) or np.around(
            result1[ind], PRECISION
        ) == np.around(
            tmp_answer, PRECISION
        ), f"value should be {tmp_answer} but got {result1[ind]}"
    # Test for weight matrix
    answer2 = [  # (contract_start_time, index, weight)
        (
            0.25,
            0,
            [
                0.0,
                0.011505841,
                0.017122296,
                0.022659486,
                0.028115871,
                0.033488173,
                0.038779225,
                0.043985932,
                0.049115842,
                0.054157988,
                0.059120574,
                0.064001158,
                0.06880285,
                0.073520932,
                0.078161722,
                0.082719578,
                0.087207934,
                0.09160591,
                0.095928687,
            ],
        ),
        (
            1.00,
            3,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.029635822,
                0.035298552,
                0.040875639,
                0.046363822,
                0.051771056,
                0.057085782,
                0.062316647,
                0.067461077,
                0.072522349,
                0.077495492,
                0.082387164,
                0.087191419,
                0.091922417,
                0.096558148,
                0.101114615,
            ],
        ),
        (
            2.00,
            7,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.061063291,
                0.067331942,
                0.073501679,
                0.079569468,
                0.085539173,
                0.09140493,
                0.097174594,
                0.102841151,
                0.108421302,
                0.113889087,
                0.119263382,
            ],
        ),
    ]
    # Test for forward swap rate
    for _, ind, tmp_answer in answer2:
        for ind2 in range(len(tmp_answer)):
            assert np.around(result2[ind, ind2], PRECISION) == np.around(
                tmp_answer[ind2], PRECISION
            ), f"value should be {tmp_answer} but got {result2[ind]}"
    print()
