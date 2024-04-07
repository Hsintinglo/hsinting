import os
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.affinity import scale


def vec_on_circle(r, p1, p2):
    p = Point(0, 0)
    c = p.buffer(r).boundary
    l = LineString([p1, p2])
    l2 = scale(l, 10000, 10000)
    i = c.intersection(l2)
    ics = [i.geoms[0].coords[0], i.geoms[1].coords[0]]
    dist2p1 = [Point(ics[0]).distance(Point(p1)), Point(ics[1]).distance(Point(p1))]
    min_ic2p1 = np.argmin(dist2p1)
    ic_real = ics[min_ic2p1]
    return ic_real

def getfns(droot, sid):
    subpaths = [os.path.join(droot, d) for d in os.listdir(droot) if os.path.isdir(os.path.join(droot, d))]
    dat_dir = [d for d in subpaths if os.path.basename(d) == sid]
    if not dat_dir:
        return None
    dat_dir = dat_dir[0]
    CONfns = [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.startswith('CON_') and f.endswith('.npy')]
    ADPfns = [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.startswith('ADP_') and f.endswith('.npy')]
    DADfns = [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.startswith('DAD_') and f.endswith('.npy')]
    return {'CONfns': CONfns, 'ADPfns': ADPfns, 'DADfns': DADfns}

def stack_data(fns, d_all, condstr, trial):
    for fn in fns:
        d = np.load(fn)
        d = pd.DataFrame(d, columns=['x', 'y', 'inflag', 'xr', 'yr', 'tarpos', 'xrc', 'yrc', 't1', 't3', 't4'])
        d['trial'] = trial
        d['condition'] = condstr
        trial += 1
        d_all = pd.concat([d_all, d])
    return d_all

# Set your root directories
droot1 = 'C:/Users/User/OneDrive/桌面/vrot analysis/visuomotor2024-master/vrot_first/data'  # Or use absolute path: 'D:/visuomotor task/visuomotor2024-master/vrot_first/data'
droot2 = 'C:/Users/User/OneDrive/桌面/vrot analysis/visuomotor2024-master/vrot_second/data' # Or use absolute path: 'D:/visuomotor task/visuomotor2024-master/vrot_second/data'

# Define your target position order
tarpos_cords = pd.DataFrame({'x': [-0.4, 0.4, 0, 0], 'y': [0, 0, 0.4, -0.4]})

# Set your list of session ids
sids = ['20240327_TJ', '20240327_PS', '20240329_MT', '20240402_YS', '20240403_TT', '20240403_MJ', '20240403_SX']

# Process each session
for sid in sids:
    proc(droot1, sid)  # Assuming proc function is defined elsewhere
