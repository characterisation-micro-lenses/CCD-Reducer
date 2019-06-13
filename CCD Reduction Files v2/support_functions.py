import numpy as np
from scipy import special
import os

# mathematical support functions
def airy(x, A, w, x0, z0):
    x = np.array(x)
    z = np.ones(x.shape)
    y = (x - x0) / w
    mask = y != 0
    z[mask] = (2 * special.j1(y[mask]) / y[mask])**2
    return A * z + z0


def gauss(x, A, sigma, x0, z0):
    """A simple Gaussian distribution with horizontal offset."""
    y = (x - x0) / sigma
    return A * np.exp(-0.5 * y**2) + z0


def moment(data, n, weights=None):
    """Calculates  moment of set. If a set of weights (they do not need to be normalized) is given they will be used as
    weights. If n=1 returns the mean, if n=2 returns the variance, otherwise returns the normalised moment.
    For the normalised moment the mean is 0 and the standard deviation is 1. Returns a float"""

    data = np.array(data)
    if weights is None:
        weights = np.ones(data.shape)
    weights = np.array(weights)
    shapetester([weights, data])
    w = weights / np.sum(weights)

    mean = np.sum(w * data)
    if n == 1:
        return mean

    var = np.sum(w * (data - mean)**2)
    if n == 2:
        return var

    std = np.sqrt(var)
    return np.sum(w * ((data - mean) / std)**n)


def refractive_index_fused_silica(wavelength):
    wl = wavelength * 1e6
    A = 0.6961663 * wl**2 / (wl**2 - 0.0684043**2)
    B = 0.4079426 * wl**2 / (wl**2 - 0.1162414**2)
    C = 0.8974794 * wl**2 / (wl**2 - 9.896161**2)
    nmin1 = A + B + C
    return np.sqrt(nmin1+1)


def Omega(D, w, err=[0, 0]):
    val = (D/w)**2
    par = np.array([D, w])
    err = np.array(err)
    error = 2 * val * np.sqrt(np.sum((err/par)**2))
    return np.array([val, error])

def S_airy(dx, NA, wl, err=[0, 0, 0]):
    dz = 0.61 * wl / NA
    par = np.array([dx, NA, wl])
    err = np.array(err)
    S = (dz / dx)**2
    error = 2*S*np.sqrt(np.sum((err/par)**2))
    return np.array([S, error])

def S_gauss(w, NA, wl, err=[0, 0, 0]):
    w_0 = wl / (np.pi * NA)
    par = np.array([w, NA, wl])
    err = np.array(err)
    S = (w_0 / w)**2
    error = 2 * S * np.sqrt(np.sum((err/par)**2))
    return np.array([S, error])



# computer science functions
def extender(x, interval=100):
    """Extend a list of numbers using linear extrapolation."""
    x0 = []
    for i in range(len(x)-1):
        if i == len(x)-2:
            x0.append(np.linspace(x[i], x[i + 1], interval + 1))
        else:
            x0.append(np.linspace(x[i], x[i + 1], interval, endpoint=False))

    return np.concatenate(x0)


def intersect(f, g):
    """Returns the indeces where an intersect happens (actually the indeces before every intersect).
    f and g can be arrays and numbers. Returns an array"""

    f, g = np.array(f).astype(np.float64), np.array(g).astype(np.float64)
    shapetester([f, g])
    idx = np.argwhere(np.diff(np.sign(f - g)) != 0).reshape(-1)
    return idx


# computer science: apertures
def aperturized(shape, position, r, sharpedge=False):
    """Creates a circular aperture and returns a copy of the data with zeros outside a given radius, the origional
    data inside the radius and half the data on the radius (if sharpedge is false)"""
    I, J = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((I - position[0])**2 + (J - position[1])**2)
    copy = np.ones(shape)
    if sharpedge is False:
        mask1 = dist <= r + 0.5
        mask2 = ((dist > r - 0.5) & (dist <= r + 0.5))
        copy[mask2] *= 0.5
    else:
        mask1 = dist <= r
    copy[np.invert(mask1)] = 0
    return copy


def aperture(shape, position, r):
    I, J = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((I - position[0])**2 + (J - position[1])**2)
    return dist <= r


# object support functions
def shapetester(arraylist):
    assert len(arraylist) != 0, "The arraylist is empty."
    sizearray = []
    for i in arraylist:
        _temp = i.shape
        if _temp != ():
            sizearray.append(_temp)

    if len(sizearray) > 0:
        assert np.all(np.equal(sizearray[0], sizearray)), "Not all sizes are equal."


def find_free_filename(path, filename, extension):
    if not does_file_exist(path, filename+str(extension)):
        return path+filename + str(extension)
    i = 1
    while True:
        if does_file_exist(path, filename+"("+str(i)+")"+str(extension)):
            i += 1
        else:
            return path+filename + "(" + str(i) + ")" + str(extension)

def does_file_exist(path, filename):
    files = sorted(os.listdir(path))
    if filename in files:
        return True
    else:
        return False
