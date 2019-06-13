def _thetaarray(a, b):
    a, b = np.array(a), np.array(b)
    assert len(a.shape) == 1
    if len(b.shape) == 1:
        shape_b = (len(b), 1)
        b = np.array([b]).transpose()
    else:
        shape_b = b.shape
    theta = np.zeros((len(a) * shape_b[0], 1 + shape_b[1]))
    for i in range(len(a)):
        theta[i * shape_b[0]:shape_b[0] * (i + 1), 0] = a[i] * np.ones(shape_b[0])
        theta[i * shape_b[0]:shape_b[0] * (i + 1), 1:] = b
    return theta


def _dearraychi2(chi2, theta, dims):
    chi2_2d = np.zeros(dims)
    for i in range(dims[0]):
        chi2_2d[i, :] = chi2[dims[1] * i:dims[1] * (i + 1)]
    return chi2_2d


def sec_to_time(s):
    hours = int(s / 3600)
    _s = s % 3600
    minutes = int(_s / 60)
    sec = int(s % 60)
    return str(hours) + ":" + str(minutes) + ":" + str(sec)
