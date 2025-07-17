import numpy as np

def in_active_volume(x, y, z):
    return abs(x) <= 30 and abs(y) <= 25 and abs(z) <= 30

def in_signal_volume(x, y, z):
    return abs(x) <= 25 and abs(y) <= 20 and abs(z) <= 25

def in_fiducial_volume(x, y, z):
    return abs(x) <= 20 and abs(y) <= 15 and abs(z) <= 20

def between(x, x_i, x_f):
    return (x >= x_i and x <= x_f) or (x <= x_i and x >= x_f)

def val(x, x_i, x_f, y_i, y_f):
    return y_i + (y_f - y_i) * ((x - x_i) / (x_f - x_i))

def get_length_in_active_volume(x_start, x_end, y_start, y_end, z_start, z_end):
    found = False
    pt = [0, 0, 0]
    
    if abs(x_start) < 30 and abs(y_start) < 25 and abs(z_start) < 30:
        found = True
        pt = [x_start, y_start, z_start]
    
    if abs(x_end) < 30 and abs(y_end) < 25 and abs(z_end) < 30:
        if found:
            return np.sqrt((x_start - x_end)**2 + (y_start - y_end)**2 + (z_start - z_end)**2)
        else:
            found = True
            pt = [x_end, y_end, z_end]

    if between(30, x_start, x_end):
        temp = [30, val(30, x_start, x_end, y_start, y_end), val(30, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= 25 and abs(temp[2]) <= 30:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-30, x_start, x_end):
        temp = [-30, val(-30, x_start, x_end, y_start, y_end), val(-30, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= 25 and abs(temp[2]) <= 30:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(25, y_start, y_end):
        temp = [val(25, y_start, y_end, x_start, x_end), 25, val(25, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) < 30 and abs(temp[2]) <= 30:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-25, y_start, y_end):
        temp = [val(-25, y_start, y_end, x_start, x_end), -25, val(-25, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) < 30 and abs(temp[2]) <= 30:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(30, z_start, z_end):
        temp = [val(30, z_start, z_end, x_start, x_end), val(30, z_start, z_end, y_start, y_end), 30]

        if abs(temp[0]) < 30 and abs(temp[1]) < 25:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-30, z_start, z_end):
        temp = [val(-30, z_start, z_end, x_start, x_end), val(-30, z_start, z_end, y_start, y_end), -30]

        if abs(temp[0]) < 30 and abs(temp[1]) < 25:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    return 0

def get_length_in_signal_volume(x_start, x_end, y_start, y_end, z_start, z_end):
    found = False
    pt = [0, 0, 0]
    
    if abs(x_start) < 25 and abs(y_start) < 20 and abs(z_start) < 25:
        found = True
        pt = [x_start, y_start, z_start]
    
    if abs(x_end) < 25 and abs(y_end) < 20 and abs(z_end) < 25:
        if found:
            return np.sqrt((x_start - x_end)**2 + (y_start - y_end)**2 + (z_start - z_end)**2)
        else:
            found = True
            pt = [x_end, y_end, z_end]

    if between(25, x_start, x_end):
        temp = [25, val(25, x_start, x_end, y_start, y_end), val(25, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= 20 and abs(temp[2]) <= 25:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-25, x_start, x_end):
        temp = [-25, val(-25, x_start, x_end, y_start, y_end), val(-25, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= 20 and abs(temp[2]) <= 25:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(20, y_start, y_end):
        temp = [val(20, y_start, y_end, x_start, x_end), 20, val(20, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) < 25 and abs(temp[2]) <= 25:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-20, y_start, y_end):
        temp = [val(-20, y_start, y_end, x_start, x_end), -20, val(-20, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) < 25 and abs(temp[2]) <= 25:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(25, z_start, z_end):
        temp = [val(25, z_start, z_end, x_start, x_end), val(25, z_start, z_end, y_start, y_end), 25]

        if abs(temp[0]) < 25 and abs(temp[1]) < 20:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-25, z_start, z_end):
        temp = [val(-25, z_start, z_end, x_start, x_end), val(-25, z_start, z_end, y_start, y_end), -25]

        if abs(temp[0]) < 25 and abs(temp[1]) < 20:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    return 0

def get_length_in_fiducial_volume(x_start, x_end, y_start, y_end, z_start, z_end):
    found = False
    pt = [0, 0, 0]
    
    if abs(x_start) < 20 and abs(y_start) < 15 and abs(z_start) < 20:
        found = True
        pt = [x_start, y_start, z_start]
    
    if abs(x_end) < 20 and abs(y_end) < 15 and abs(z_end) < 20:
        if found:
            return np.sqrt((x_start - x_end)**2 + (y_start - y_end)**2 + (z_start - z_end)**2)
        else:
            found = True
            pt = [x_end, y_end, z_end]

    if between(20, x_start, x_end):
        temp = [20, val(20, x_start, x_end, y_start, y_end), val(20, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= 15 and abs(temp[2]) <= 20:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-20, x_start, x_end):
        temp = [-20, val(-20, x_start, x_end, y_start, y_end), val(-20, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= 15 and abs(temp[2]) <= 20:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(15, y_start, y_end):
        temp = [val(15, y_start, y_end, x_start, x_end), 15, val(15, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) < 20 and abs(temp[2]) <= 20:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-15, y_start, y_end):
        temp = [val(-15, y_start, y_end, x_start, x_end), -15, val(-15, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) < 20 and abs(temp[2]) <= 20:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(20, z_start, z_end):
        temp = [val(20, z_start, z_end, x_start, x_end), val(20, z_start, z_end, y_start, y_end), 20]

        if abs(temp[0]) < 20 and abs(temp[1]) < 15:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    if between(-20, z_start, z_end):
        temp = [val(-20, z_start, z_end, x_start, x_end), val(-20, z_start, z_end, y_start, y_end), -20]

        if abs(temp[0]) < 20 and abs(temp[1]) < 15:
            if found:
                return np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
            else:
                found = True
                pt = temp

    return 0