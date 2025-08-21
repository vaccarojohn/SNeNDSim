import numpy as np

def in_active_volume(x, y, z):
    return abs(x) <= 30 and abs(y) <= 25 and abs(z) <= 30

def in_signal_volume(x, y, z):
    return abs(x) <= 25 and abs(y) <= 20 and abs(z) <= 25

def in_fiducial_volume(x, y, z):
    return abs(x) <= 20 and abs(y) <= 15 and abs(z) <= 20

def between(x, x_i, x_f):
    return (x > x_i and x < x_f) or (x < x_i and x > x_f)

def val(x, x_i, x_f, y_i, y_f):
    return y_i + (y_f - y_i) * ((x - x_i) / (x_f - x_i))
    
def get_length_in_box(x_start, x_end, y_start, y_end, z_start, z_end, xbox, ybox, zbox):
    found = False
    pt = [0, 0, 0]
    
    if abs(x_start) <= xbox and abs(y_start) <= ybox and abs(z_start) <= zbox:
        found = True
        pt = [x_start, y_start, z_start]
    
    if abs(x_end) <= xbox and abs(y_end) <= ybox and abs(z_end) <= zbox:
        if found:
            return np.sqrt((x_start - x_end)**2 + (y_start - y_end)**2 + (z_start - z_end)**2)
        else:
            found = True
            pt = [x_end, y_end, z_end]

    if between(xbox, x_start, x_end):
        temp = [xbox, val(xbox, x_start, x_end, y_start, y_end), val(xbox, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= ybox and abs(temp[2]) <= zbox:
            if found:
                d = np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
                if d != 0:
                    return d
            else:
                found = True
                pt = temp

    if between(-xbox, x_start, x_end):
        temp = [-xbox, val(-xbox, x_start, x_end, y_start, y_end), val(-xbox, x_start, x_end, z_start, z_end)]

        if abs(temp[1]) <= ybox and abs(temp[2]) <= zbox:
            if found:
                d = np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
                if d != 0:
                    return d
            else:
                found = True
                pt = temp

    if between(ybox, y_start, y_end):
        temp = [val(ybox, y_start, y_end, x_start, x_end), ybox, val(ybox, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) <= xbox and abs(temp[2]) <= zbox:
            if found:
                d = np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
                if d != 0:
                    return d
            else:
                found = True
                pt = temp

    if between(-ybox, y_start, y_end):
        temp = [val(-ybox, y_start, y_end, x_start, x_end), -ybox, val(-ybox, y_start, y_end, z_start, z_end)]

        if abs(temp[0]) <= xbox and abs(temp[2]) <= zbox:
            if found:
                d = np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
                if d != 0:
                    return d
            else:
                found = True
                pt = temp

    if between(zbox, z_start, z_end):
        temp = [val(zbox, z_start, z_end, x_start, x_end), val(zbox, z_start, z_end, y_start, y_end), zbox]

        if abs(temp[0]) <= xbox and abs(temp[1]) <= ybox:
            if found:
                d = np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
                if d != 0:
                    return d
            else:
                found = True
                pt = temp

    if between(-zbox, z_start, z_end):
        temp = [val(-zbox, z_start, z_end, x_start, x_end), val(-zbox, z_start, z_end, y_start, y_end), -zbox]

        if abs(temp[0]) <= xbox and abs(temp[1]) <= ybox:
            if found:
                d = np.sqrt((temp[0] - pt[0])**2 + (temp[1] - pt[1])**2 + (temp[2] - pt[2])**2)
                if d != 0:
                    return d
            else:
                found = True
                pt = temp

    return 0

def get_length_in_active_volume(x_start, x_end, y_start, y_end, z_start, z_end):
    return get_length_in_box(x_start, x_end, y_start, y_end, z_start, z_end, 30, 25, 30)

def get_length_in_signal_volume(x_start, x_end, y_start, y_end, z_start, z_end):
    return get_length_in_box(x_start, x_end, y_start, y_end, z_start, z_end, 25, 20, 25)

def get_length_in_fiducial_volume(x_start, x_end, y_start, y_end, z_start, z_end):
    return get_length_in_box(x_start, x_end, y_start, y_end, z_start, z_end, 20, 15, 20)

def get_length_in_cosmic_ray_taggers(x_start, x_end, y_start, y_end, z_start, z_end):
    CRTtop = get_length_in_box(x_start, x_end, y_start, y_end, z_start - 62, z_end - 62, 50, 50, 1.5)
    CRTbottom = get_length_in_box(x_start, x_end, y_start, y_end, z_start + 62, z_end + 62, 50, 50, 1.5)
    CRTleft = get_length_in_box(x_start + 52, x_end + 52, y_start, y_end, z_start, z_end, 1.5, 50, 60)
    CRTright = get_length_in_box(x_start - 52, x_end - 52, y_start, y_end, z_start, z_end, 1.5, 50, 60)
    CRTfront = get_length_in_box(x_start, x_end, y_start + 52, y_end + 52, z_start, z_end, 50, 1.5, 60)
    CRTback = get_length_in_box(x_start, x_end, y_start - 52, y_end - 52, z_start, z_end, 50, 1.5, 60)

    return (CRTtop, CRTbottom, CRTleft, CRTright, CRTfront, CRTback)
    