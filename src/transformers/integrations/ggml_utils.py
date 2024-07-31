# Constants for ggml imatrix dequantization
# migrate from https://github.com/ggerganov/ggml/blob/3f5a4bbe59285c0f679b376f6259187d5514ff9c/src/ggml-common.h#L437
import numpy as np


IQ2XXS_GRID = np.array(
    [
        0x0808080808080808,
        0x080808080808082B,
        0x0808080808081919,
        0x0808080808082B08,
        0x0808080808082B2B,
        0x0808080808190819,
        0x0808080808191908,
        0x08080808082B0808,
        0x08080808082B082B,
        0x08080808082B2B08,
        0x08080808082B2B2B,
        0x0808080819080819,
        0x0808080819081908,
        0x0808080819190808,
        0x0808080819192B08,
        0x08080808192B0819,
        0x08080808192B1908,
        0x080808082B080808,
        0x080808082B08082B,
        0x080808082B082B2B,
        0x080808082B2B082B,
        0x0808081908080819,
        0x0808081908081908,
        0x0808081908190808,
        0x0808081908191919,
        0x0808081919080808,
        0x080808192B081908,
        0x080808192B192B08,
        0x0808082B08080808,
        0x0808082B0808082B,
        0x0808082B082B082B,
        0x0808082B2B08082B,
        0x0808190808080819,
        0x0808190808081908,
        0x0808190808190808,
        0x08081908082B0819,
        0x08081908082B1908,
        0x0808190819080808,
        0x080819081908082B,
        0x0808190819082B08,
        0x08081908192B0808,
        0x080819082B080819,
        0x080819082B081908,
        0x080819082B190808,
        0x080819082B2B1908,
        0x0808191908080808,
        0x080819190808082B,
        0x0808191908082B08,
        0x08081919082B0808,
        0x080819191908192B,
        0x08081919192B2B19,
        0x080819192B080808,
        0x080819192B190819,
        0x0808192B08082B19,
        0x0808192B08190808,
        0x0808192B19080808,
        0x0808192B2B081908,
        0x0808192B2B2B1908,
        0x08082B0808080808,
        0x08082B0808081919,
        0x08082B0808082B08,
        0x08082B0808191908,
        0x08082B08082B2B08,
        0x08082B0819080819,
        0x08082B0819081908,
        0x08082B0819190808,
        0x08082B081919082B,
        0x08082B082B082B08,
        0x08082B1908081908,
        0x08082B1919080808,
        0x08082B2B0808082B,
        0x08082B2B08191908,
        0x0819080808080819,
        0x0819080808081908,
        0x0819080808190808,
        0x08190808082B0819,
        0x0819080819080808,
        0x08190808192B0808,
        0x081908082B081908,
        0x081908082B190808,
        0x081908082B191919,
        0x0819081908080808,
        0x0819081908082B08,
        0x08190819082B0808,
        0x0819081919190808,
        0x0819081919192B2B,
        0x081908192B080808,
        0x0819082B082B1908,
        0x0819082B19081919,
        0x0819190808080808,
        0x0819190808082B08,
        0x08191908082B0808,
        0x08191908082B1919,
        0x0819190819082B19,
        0x081919082B080808,
        0x0819191908192B08,
        0x08191919192B082B,
        0x0819192B08080808,
        0x0819192B0819192B,
        0x08192B0808080819,
        0x08192B0808081908,
        0x08192B0808190808,
        0x08192B0819080808,
        0x08192B082B080819,
        0x08192B1908080808,
        0x08192B1908081919,
        0x08192B192B2B0808,
        0x08192B2B19190819,
        0x082B080808080808,
        0x082B08080808082B,
        0x082B080808082B2B,
        0x082B080819081908,
        0x082B0808192B0819,
        0x082B08082B080808,
        0x082B08082B08082B,
        0x082B0819082B2B19,
        0x082B081919082B08,
        0x082B082B08080808,
        0x082B082B0808082B,
        0x082B190808080819,
        0x082B190808081908,
        0x082B190808190808,
        0x082B190819080808,
        0x082B19081919192B,
        0x082B191908080808,
        0x082B191919080819,
        0x082B1919192B1908,
        0x082B192B2B190808,
        0x082B2B0808082B08,
        0x082B2B08082B0808,
        0x082B2B082B191908,
        0x082B2B2B19081908,
        0x1908080808080819,
        0x1908080808081908,
        0x1908080808190808,
        0x1908080808192B08,
        0x19080808082B0819,
        0x19080808082B1908,
        0x1908080819080808,
        0x1908080819082B08,
        0x190808081919192B,
        0x19080808192B0808,
        0x190808082B080819,
        0x190808082B081908,
        0x190808082B190808,
        0x1908081908080808,
        0x19080819082B0808,
        0x19080819192B0819,
        0x190808192B080808,
        0x190808192B081919,
        0x1908082B08080819,
        0x1908082B08190808,
        0x1908082B19082B08,
        0x1908082B1919192B,
        0x1908082B192B2B08,
        0x1908190808080808,
        0x1908190808082B08,
        0x19081908082B0808,
        0x190819082B080808,
        0x190819082B192B19,
        0x190819190819082B,
        0x19081919082B1908,
        0x1908192B08080808,
        0x19082B0808080819,
        0x19082B0808081908,
        0x19082B0808190808,
        0x19082B0819080808,
        0x19082B0819081919,
        0x19082B1908080808,
        0x19082B1919192B08,
        0x19082B19192B0819,
        0x19082B192B08082B,
        0x19082B2B19081919,
        0x19082B2B2B190808,
        0x1919080808080808,
        0x1919080808082B08,
        0x1919080808190819,
        0x1919080808192B19,
        0x19190808082B0808,
        0x191908082B080808,
        0x191908082B082B08,
        0x1919081908081908,
        0x191908191908082B,
        0x191908192B2B1908,
        0x1919082B2B190819,
        0x191919082B190808,
        0x191919082B19082B,
        0x1919191908082B2B,
        0x1919192B08080819,
        0x1919192B19191908,
        0x19192B0808080808,
        0x19192B0808190819,
        0x19192B0808192B19,
        0x19192B08192B1908,
        0x19192B1919080808,
        0x19192B2B08082B08,
        0x192B080808081908,
        0x192B080808190808,
        0x192B080819080808,
        0x192B0808192B2B08,
        0x192B081908080808,
        0x192B081919191919,
        0x192B082B08192B08,
        0x192B082B192B0808,
        0x192B190808080808,
        0x192B190808081919,
        0x192B191908190808,
        0x192B19190819082B,
        0x192B19192B081908,
        0x192B2B081908082B,
        0x2B08080808080808,
        0x2B0808080808082B,
        0x2B08080808082B2B,
        0x2B08080819080819,
        0x2B0808082B08082B,
        0x2B08081908081908,
        0x2B08081908192B08,
        0x2B08081919080808,
        0x2B08082B08190819,
        0x2B08190808080819,
        0x2B08190808081908,
        0x2B08190808190808,
        0x2B08190808191919,
        0x2B08190819080808,
        0x2B081908192B0808,
        0x2B08191908080808,
        0x2B0819191908192B,
        0x2B0819192B191908,
        0x2B08192B08082B19,
        0x2B08192B19080808,
        0x2B08192B192B0808,
        0x2B082B080808082B,
        0x2B082B1908081908,
        0x2B082B2B08190819,
        0x2B19080808081908,
        0x2B19080808190808,
        0x2B190808082B1908,
        0x2B19080819080808,
        0x2B1908082B2B0819,
        0x2B1908190819192B,
        0x2B1908192B080808,
        0x2B19082B19081919,
        0x2B19190808080808,
        0x2B191908082B082B,
        0x2B19190819081908,
        0x2B19191919190819,
        0x2B192B082B080819,
        0x2B192B19082B0808,
        0x2B2B08080808082B,
        0x2B2B080819190808,
        0x2B2B08082B081919,
        0x2B2B081908082B19,
        0x2B2B082B08080808,
        0x2B2B190808192B08,
        0x2B2B2B0819190808,
        0x2B2B2B1908081908,
    ]
)

KSIGNS_IQ2XS = np.array(
    [
        0,
        129,
        130,
        3,
        132,
        5,
        6,
        135,
        136,
        9,
        10,
        139,
        12,
        141,
        142,
        15,
        144,
        17,
        18,
        147,
        20,
        149,
        150,
        23,
        24,
        153,
        154,
        27,
        156,
        29,
        30,
        159,
        160,
        33,
        34,
        163,
        36,
        165,
        166,
        39,
        40,
        169,
        170,
        43,
        172,
        45,
        46,
        175,
        48,
        177,
        178,
        51,
        180,
        53,
        54,
        183,
        184,
        57,
        58,
        187,
        60,
        189,
        190,
        63,
        192,
        65,
        66,
        195,
        68,
        197,
        198,
        71,
        72,
        201,
        202,
        75,
        204,
        77,
        78,
        207,
        80,
        209,
        210,
        83,
        212,
        85,
        86,
        215,
        216,
        89,
        90,
        219,
        92,
        221,
        222,
        95,
        96,
        225,
        226,
        99,
        228,
        101,
        102,
        231,
        232,
        105,
        106,
        235,
        108,
        237,
        238,
        111,
        240,
        113,
        114,
        243,
        116,
        245,
        246,
        119,
        120,
        249,
        250,
        123,
        252,
        125,
        126,
        255,
    ]
)

KMASK_IQ2XS = np.array([1, 2, 4, 8, 16, 32, 64, 128])
