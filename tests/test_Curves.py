from Curves import *


def test_generate_caplet_vol_term_structure_black_vol():
    PRECISION = 5

    # Cap prices
    cap_prices = [
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

    vol_curve = Vol_curve(
        cap_prices, zcb_curve, tenors, interp_method="piecewise constant"
    )
    vol_curve.generate_caplet_vol_term_structure()

    # Cap vol
    answer = [
        0.2497,
        0.2497,
        0.2497,
        0.301687537,
        0.328146146,
        0.355832179,
        0.385158652,
        0.341589224,
        0.349360359,
        0.357135024,
        0.364899952,
        0.341235359,
        0.342603534,
        0.344035234,
        0.345370083,
        0.333442068,
        0.332617638,
        0.332124396,
        0.331388131,
    ]

    answer = np.around(answer, PRECISION)
    result = np.around(vol_curve.caplet_black_vols, PRECISION)

    for i in range(len(answer)):
        assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
            i
        ], f"value should be {answer} but got {result}"


# def test_generate_caplet_vol_term_structure_normal_vol():
#     PRECISION = 5

#     cap_prices = [
#         0.0004087808446,
#         0.001172267395,
#         0.002143362268,
#         0.003249245946,
#         0.004464872907,
#         0.005811949117,
#         0.00741885408,
#         0.009259999071,
#     ]

#     tenors = [
#         0.25,
#         0.50,
#         0.75,
#         1.00,
#         1.25,
#         1.50,
#         1.75,
#         2.00,
#         2.25,
#     ]

#     zcb_curve = [
#         0.987691146,
#         0.977440241,
#         0.968781995,
#         0.96109015,
#         0.953984871,
#         0.947146003,
#         0.940369932,
#         0.933576529,
#         0.926825762,
#     ]

#     vol_curve = Vol_curve(
#         cap_prices, zcb_curve, tenors, interp_method="piecewise constant"
#     )
#     vol_curve.generate_caplet_vol_term_structure()
#     answer = [
#         0.00838649,
#         0.008461329,
#         0.008583391,
#         0.008799824,
#         0.009116849,
#         0.009525188,
#         0.010157405,
#         0.010811121,
#     ]

#     answer = np.around(answer, PRECISION)
#     result = np.around(vol_curve.caplet_normal_vols, PRECISION)

#     for i in range(len(answer)):
#         assert (np.isnan(result[i]) and np.isnan(answer[i])) or result[i] == answer[
#             i
#         ], f"value should be {answer} but got {result}"


if __name__ == "__main__":
    test_generate_caplet_vol_term_structure_black_vol()
    # test_generate_caplet_vol_term_structure_normal_vol()
    print("---")
