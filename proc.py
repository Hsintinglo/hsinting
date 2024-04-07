import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
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

def comp_angle(u, v):
    if any(np.isnan(u)) or any(np.isnan(v)):
        return np.nan
    theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
    theta = np.degrees(theta)
    if theta > 180:
        theta = 360 - theta
    return theta

def proc(droot1, droot2, sid):
    info1 = getfns(droot1, sid)
    info2 = getfns(droot2, sid)
    
    d_all = pd.DataFrame()
    trial = 1
    
    d_all = stack_data(info1['CONfns'], d_all, 'CON', trial)
    trial = max(d_all['trial']) + 1
    d_all = stack_data(info1['ADPfns'], d_all, 'ADP', trial)
    trial = max(d_all['trial']) + 1
    d_all = stack_data(info2['ADPfns'], d_all, 'ADP', trial)
    trial = max(d_all['trial']) + 1
    d_all = stack_data(info2['DADfns'], d_all, 'DAD', trial)
    
    temp = pd.DataFrame({'trial': d_all['trial'].unique()})
    temp['errblock'] = pd.qcut(temp['trial'], 72, labels=False) + 1
    temp['timeblock'] = pd.qcut(temp['trial'], 12, labels=False) + 1
    
    d_all = d_all.merge(temp, on='trial', how='left')
    
    temp = d_all[['trial', 'inflag']].groupby('trial').cumcount() + 1
    d_all['sample_id'] = temp
    temp['min_seg'] = temp.groupby(['trial', 'inflag'])['sample_id'].transform('min')
    
    temp_count = temp.groupby('trial').size().reset_index(name='n').query('n > 1')
    
    temp_w = temp.query('trial in @temp_count.trial').pivot_table(index='trial', columns='inflag', values='min_seg', aggfunc='first').add_prefix('inflag_min_').reset_index()
    
    # 其余部分暂未转换
    
    return d_cross_bk

