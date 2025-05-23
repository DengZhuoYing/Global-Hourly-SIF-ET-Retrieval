from numpy import pi, tan, cos, sin
from numpy import arctan, sqrt, arccos
import numpy as np


def rtlsReflectance(Kiso, Kgeo, Kvol, sza, vza, saa, vaa, raa=None):
    h_b = 2.0
    b_r = 1.0

    if raa is None:
        raa = vaa - saa

    raa = raa * pi / 180.0
    sza = sza * pi / 180.0
    vza = vza * pi / 180.0

    Riso = 1.0

    szaP = arctan(b_r * tan(sza))
    vzaP = arctan(b_r * tan(vza))
    cos_thetaP = cos(szaP) * cos(vzaP) + sin(szaP) * sin(vzaP) * cos(raa)
    dP = sqrt(tan(vzaP) * tan(vzaP) + tan(szaP) * tan(szaP) - 2.0 * tan(vzaP) * tan(szaP) * cos(raa))
    amfP = (1. / cos(vzaP)) + (1. / cos(szaP))
    temp = h_b * sqrt(dP * dP + (tan(szaP) * tan(vzaP) * sin(raa)) * (tan(szaP) * tan(vzaP) * sin(raa))) / amfP
    cos_t = np.array([min([1, t]) for t in temp])
    t = arccos(cos_t)
    sin_t = sin(t)
    O = (1.0 - (1. / pi) * (t - sin_t * cos_t)) * amfP

    Rgeo = 0.5 * (1 + cos_thetaP) * (1. / cos(vzaP)) * (1. / cos(szaP)) - O

    cos_theta = cos(sza) * cos(vza) + sin(sza) * sin(vza) * cos(raa)
    theta = arccos(cos_theta)

    Rvol = (((0.5 * pi - theta) * cos_theta + sin(theta)) / (cos(sza) + cos(vza))) - 0.25 * pi

    rho = Kiso * Riso + Kgeo * Rgeo + Kvol * Rvol

    return rho
