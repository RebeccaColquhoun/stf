import numpy as np
import math

list_moments = np.logspace(9, 23, 15)
pi = math.pi
c = (3 * pi)/16

def convert_mag_to_moment(magntiude):
    """
    Convert magnitude to moment
    """

    moment = 10 ** (1.5 * magntiude + 9.1)

    return moment

def convert_moment_to_mag(moment):
    """
    Convert moment to magnitude
    """

    magnitude = (2/3) * (math.log10(moment) - 9.1)

    return magnitude

def calc_duration(moment, V = 2500, stressdrop = 5E6, w_max = 60E3):
    if moment < c * stressdrop * w_max ** 3:
        logT = 1 / 3 * np.log10(moment) - 1/3 * np.log10(c * stressdrop * V ** 3)
    else:
        logT = np.log10(moment)-np.log10(c*stressdrop*(w_max**2)*V)
    return 10**logT


def calc_moment(duration, V = 2500, stressdrop = 5E6, w_max = 60E3):
    logT = np.log10(duration)
    if logT < 1/3*np.log10(c*stressdrop*w_max**3) - 1/3*np.log10(c*stressdrop*V**3):
        moment = 10**(3*logT+np.log10(c*stressdrop*V**3))
    else:
        moment = 10**(logT+np.log10(c*stressdrop*(w_max**2)*V))
    return moment
