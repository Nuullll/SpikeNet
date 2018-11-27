class NetworkConfig(object):
    # simulation resolution (s)
    DT = 0.001
    DT_MS = 1000 * DT

    # layer size
    INPUT_IMAGE_WIDTH = 11
    INPUT_IMAGE_HEIGHT = 11
    INPUT_NEURON_NUMBER = INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT
    OUTPUT_NEURON_NUMBER = 60

    # duration per stimuli (ms)
    DURATION_PER_TRAINING_IMAGE = 100
    DURATION_PER_TESTING_IMAGE = 100

    # average firing rate of input neurons (Hz)
    AVERAGE_INPUT_FIRING_RATE = 40.


class LayerConfig(object):
    O_REST = 0.
    O_PEAK = 1.


class LIFLayerConfig(LayerConfig):
    V_REST = 0.
    V_TH_REST = 1.
    DV_TH = 0.03
    LEAK_FACTOR = 0.1
    REFRACTORY = 0
    RES = 2.
    WINNERS = 4


class ConnectionConfig(object):
    learn_factor = 1
    decay_factor = 0.264
    LEARN_RATE_P = 5e-4 * learn_factor
    LEARN_RATE_M = 5e-4 * learn_factor
    TAU_P = 20
    TAU_M = 20
    DECAY = 1e-4 * decay_factor
    W_MIN = 0.2
    W_MAX = 1.
