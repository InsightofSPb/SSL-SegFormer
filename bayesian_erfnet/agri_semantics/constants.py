class ActiveLearners:
    RANDOM = "random"
    BALD = "bald"
    ALL = "all"

class Models:
    ERFNET = "erfnet"
    BAYESIAN_ERFNET = "bayesian_erfnet"
    EVIDENTIAL_ERFNET = "evidential_erfnet"
    UNET = "unet"
    SegFormer = "SegFormer"


class Losses:
    CROSS_ENTROPY = "xentropy"
    SOFT_IOU = "soft_iou"
    MSE = "mse"
    PAC_TYPE_2_MLE = "pac_type_2_mle"
    CROSS_ENTROPY_BAYES_RISK = "xentropy_bayes_risk"
    MSE_BAYES_RISK = "mse_bayes_risk"


class Maps:
    # Federico original
    # MERGE = {
    #     0:0,  # ignore?
    #     1:1,
    #     2:2,
    #     3:3,
    #     4:1,
    #     5:0,
    #     6:0,
    #     7:0,
    #     8:0,
    #     9:0,
    #     10:0,
    #     11:0,
    #     12:12,
    #     13:13,
    #     14:13,
    #     15:12,
    #     16:16,
    #     17:16,
    #     18:1,
    #     255:0
    # }

    # CONTIGUOS = {
    #     0:0,
    #     1:1,
    #     2:2,
    #     3:3,
    #     12:4,
    #     13:5,
    #     16:6
    # }

    MERGE = {
        0: 0,  # ignore?
        1: 1,
        2: 2,
        3: 3,
        4: 1,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 12,
        13: 2,
        14: 2,
        15: 12,
        16: 16,
        17: 16,
        18: 1,
        255: 0,
    }

    CONTIGUOS = {0: 0, 1: 1, 2: 2, 3: 3, 12: 4, 16: 5}
