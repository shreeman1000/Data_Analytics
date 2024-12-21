import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from math import sin, cos, tan, sqrt, pi
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def quad_root(a, b, c):
    root1 = (-b + sqrt(b**2 - 4 * a * c)) / (2 * a)
    root2 = (-b - sqrt(b**2 - 4 * a * c)) / (2 * a)

    return [root1, root2]

def Plotting(c, e1, e2, z, r, s, times, oppositions):

    Ex = e1 * cos(e2) 
    Ey = e1 * sin(e2)
    Cx = cos(c)
    Cy = sin(c)
    Z = (z + s * times) % (2 * pi)

    calcXs = [Ex]
    calcYs = [Ey]
    obsXs = [0]
    obsYs = [0]

    plt.figure(figsize=(10,10))
    circle = plt.Circle((Cx, Cy), r, color='red', fill=False, ec="black")

    for i in range(len(Z)):
        K_line = Ey - tan(Z[i]) * Ex
        a = (1 + tan(Z[i])**2)
        b = -2 * Cx + 2 * tan(Z[i]) * (Ey - tan(Z[i]) * Ex - Cy)
        c = Cx**2 + (Ey - tan(Z[i]) * Ex - Cy)**2 - r**2

        if b**2 < 4 * a * c:
            break
        Root = quad_root(a, b, c)

        if Z[i] < 3 * pi / 2 and Z[i] > pi / 2:
            Rx = Root[1]
        else:
            Rx = Root[0]

        Ry = tan(Z[i]) * Rx + K_line

        calcXs.append(Rx)
        calcYs.append(Ry)

    for i in range(len(oppositions)):
        a = 1 + tan(np.radians(oppositions[i]))**2
        b = -2 * Cx - 2 * Cy * tan(np.radians(oppositions[i]))
        c = Cx**2 + Cy**2 - r**2

        Root = quad_root(a, b, c)

        if np.radians(oppositions[i]) < 3 * pi / 2 and np.radians(
                oppositions[i]) > pi / 2:
            Rx = Root[1]
        else:
            Rx = Root[0]

        Ry = tan(np.radians(oppositions[i])) * Rx

        obsXs.append(Rx)
        obsYs.append(Ry)

    ax = plt.gca()
    for i in range(1, len(calcXs)):
        plt.plot([calcXs[i], calcXs[0]], [calcYs[i], calcYs[0]],
                    'o--',
                    markersize=5,
                    mfc='black',
                    mec='black')
        plt.plot([obsXs[i], obsXs[0]], [obsYs[i], obsYs[0]],
                    'o-',
                    markersize=5,
                    mfc='black',
                    mec='black')

    ax.add_patch(circle)
    plt.show()
    return None

def MarsEquantModel(c, r, e1, e2, z, s, times, oppositions):

    Ex = e1 * cos(e2)
    Ey = e1 * sin(e2)
    Cx = cos(c)
    Cy = sin(c) 
    error = []
    Z = (z + s * times) % (2 * pi) 

    for i in range(len(Z)):
        K_line = Ey - tan(Z[i]) * Ex

        a = (1 + tan(Z[i])**2)
        b = -2 * Cx + 2 * tan(Z[i]) * (Ey - tan(Z[i]) * Ex - Cy)
        c = Cx**2 + (Ey - tan(Z[i]) * Ex - Cy)**2 - r**2

        if b**2 < 4 * a * c:
            break

        Root = quad_root(a, b, c)

        if 3 * pi / 2 > Z[i] > pi / 2:
            Rx = Root[1]
        else:
            Rx = Root[0]

        Ry = tan(Z[i]) * Rx + K_line
        Calc_long = ((np.arctan2(float(Ry), float(Rx)) + 4 * pi) % (2 * pi)) * 180 / pi

        error.append(Calc_long - oppositions[i])

    error = np.array(error)
    error = (error + 720) % 360

    for x in range(len(error)):
        if error[x] > 180:
            error[x] = error[x] - 360
        elif error[x] < -180:
            error[x] = error[x] + 360

    if len(error) == 0:
        maxError = 10000
    else:
        maxError = max(np.abs(np.array(error)))

    return error, maxError

def bestc(c, e1, e2, z, r, s, times, oppositions):
    _, maxError = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    return maxError

def beste1(e1, c, e2, z, r, s, times, oppositions):
    _, maxError = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    return maxError

def beste2(e2, c, e1, z, r, s, times, oppositions):
    _, maxError = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    return maxError

def bestz(z, c, e1, e2, r, s, times, oppositions):
    _, maxError = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    return maxError

def MAXERROR_net(Vars, r, s, times, oppositions):
    (c, e1, e2, z) = Vars
    error, maxError = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    return maxError

def bestOrbitInnerParams(r, s, times, oppositions):
    c = np.radians(100)
    z = np.radians(60)
    e2 = np.radians(70)
    e1 = 1

    res = minimize_scalar(beste1,
                            e1,
                            args=(c, e2, z, r, s, times, oppositions),
                            method='Bounded',
                            options={
                                'disp': False,
                                'maxiter': 20000
                            },
                            bounds=(0.9, 1.7))
    e1 = res.x

    res = minimize_scalar(bestz,
                            z,
                            args=(c, e1, e2, r, s, times, oppositions),
                            method='Bounded',
                            options={
                                'disp': False,
                                'maxiter': 20000
                            },
                            bounds=(-pi, pi))
    z = res.x

    res = minimize_scalar(bestc,
                            c,
                            args=(e1, e2, z, r, s, times, oppositions),
                            method='Bounded',
                            options={
                                'disp': False,
                                'maxiter': 20000
                            },
                            bounds=(-pi, pi))
    c = res.x

    res = minimize_scalar(beste2,
                            e2,
                            args=(c, e1, z, r, s, times, oppositions),
                            method='Bounded',
                            options={
                                'disp': False,
                                'maxiter': 20000
                            },
                            bounds=(-pi, pi))
    e2 = res.x

    res = minimize(MAXERROR_net, (c, e1, e2, z),
                    args=(r, s, times, oppositions),
                    method='Nelder-Mead',
                    options={
                        'disp': False,
                        'maxiter': 20000
                    },
                    bounds=((-pi, pi), (0.9, 1.7), (-pi, pi), (-pi, pi)))
    c, e1, e2, z = res.x[0], res.x[1], res.x[2], res.x[3]

    error, maxError = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)

    return c, e1, e2, z, error, maxError

def bestS(r, times, oppositions):

    S = np.arange((1 - 0.2) * 2 * pi / 687, (1 + 0.2) * 2 * pi / 687,
                    pi / 100000)
    maxError = 10000
    error = []
    for test_s in range(len(S)):
        c, e1, e2, z, err, maxErr = bestOrbitInnerParams(
            r, S[test_s], times, oppositions)

        if maxError > maxErr:
            maxError = maxErr
            error = err
            best_s = S[test_s]

    return best_s, error, maxError

def bestR(s, times, oppositions):

    R = np.arange(7.5, 9, 0.005)
    error = []
    maxError = 10000
    for test_r in range(len(R)):

        c, e1, e2, z, err, maxErr = bestOrbitInnerParams(
            R[test_r], s, times, oppositions)

        if maxError > maxErr:
            maxError = maxErr
            error = err
            best_r = R[test_r]

    return best_r, error, maxError

def bestMarsOrbitParams(times, oppositions):

    BEST_S = 2 * pi / 687
    BEST_R = 10
    maxError = 10000
    error = []

    for r_test in np.arange(7.5, 9, 0.005):
        for s_test in np.arange((1 - 0.002) * 2 * pi / 687, (1 + 0.002) * 2 * pi / 687, pi / 10000000):
            c, e1, e2, z, err, maxErr = bestOrbitInnerParams(
                r_test, s_test, times, oppositions)
            if maxErr < maxError:
                maxError = maxErr
                error = err
                goodVar = (c, e1, e2, z)
                BEST_R = r_test
                BEST_S = s_test

    (c, e1, e2, z) = (goodVar[0], goodVar[1], goodVar[2], goodVar[3])

    _, maxError = MarsEquantModel(c, BEST_R, e1, e2, z, BEST_S, times, oppositions)
    Plotting(c, e1, e2, z, BEST_R, BEST_S, times, oppositions)
    (c, e1, e2, z) = (goodVar[0]*180/pi, goodVar[1], goodVar[2]*180/pi, goodVar[3]*180/pi)

    return BEST_R, BEST_S, c, e1, e2, z, error, maxError

def get_times(mars):
    times = [0]
    for i in range(1, 12):
        times.append(
            (dt.datetime(mars['Year'][i], mars['Month'][i], mars['Day'][i],
                         mars['Hour'][i], mars['Minute'][i]) -
             dt.datetime(mars['Year'][i - 1], mars['Month'][i - 1],
                         mars['Day'][i - 1], mars['Hour'][i - 1],
                         mars['Minute'][i - 1])).total_seconds() / (24 * 3600))
        
    mars['times'] = times
    mars['times'] = mars['times'].cumsum()
    times = np.array(mars['times'])
    return times

def get_oppositions(mars):
    mars['oppositions'] = (mars["ZodiacIndex"]) * 30 + mars[
        'Degree'] + mars['Minute.1'] / 60 + mars['Second'] / 3600
    
    oppositions = np.array(mars['oppositions'])

    return oppositions

if __name__ == "__main__":

    mars = pd.read_csv(
        "01_data_mars_opposition_updated.csv"
    )

    times = get_times(mars)

    oppositions = get_oppositions(mars)

    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(times, oppositions)

    print("Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(r, s, c, e1, e2, z))
    print("The maximum angular error = {:2.4f}".format(maxError))
    print(errors)