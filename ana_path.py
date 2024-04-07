import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Define functions

# Compute signed angle
def comp_angle(u, v):
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return np.nan
    theta = np.rad2deg(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]))
    if theta > 180:
        theta = 360 - theta
    return theta

# Get file names
def getfns(droot, sid):
    dat_dir = os.path.join(droot, sid)
    print(dat_dir)
    CONfns = glob(os.path.join(dat_dir, 'CON_*.npy'))
    ADPfns = glob(os.path.join(dat_dir, 'ADP_*.npy'))
    DADfns = glob(os.path.join(dat_dir, 'DAD_*.npy'))
    return {'CONfns': CONfns, 'ADPfns': ADPfns, 'DADfns': DADfns}

# Stack data
def stack_data(fns, condstr, trial):
    d_all = pd.DataFrame()
    for fn in fns:
        d = np.load(fn)
        d = pd.DataFrame(d, columns=['x', 'y', 'inflag', 'xr', 'yr', 'tarpos', 'xrc', 'yrc', 't1', 't3', 't4'])
        d['trial'] = trial
        d['condition'] = condstr
        trial += 1
        d_all = pd.concat([d_all, d], ignore_index=True)
    return d_all

# Define target positions
tarpos_cords = pd.DataFrame({'x': [-0.4, 0.4, 0, 0], 'y': [0, 0, 0.4, -0.4]})

# Define data root directories
droot1 = C:/Users/User/桌面/vrot analysis/visuomotor2024-master/vrot_first/data #'../vrot_first/data'
droot2 = C:/Users/User/桌面/vrot analysis/visuomotor2024-master/vrot_second/data #'../vrot_second/data'

# List of sessions
sids = ['20240327_TJ', '20240327_PS', '20240329_MT', '20240402_YS', '20240403_TT', '20240403_MJ', '20240403_SX']

for sid in sids:
    print(f"Now processing {sid} ...")
    info1 = getfns(droot1, sid)
    info2 = getfns(droot2, sid)

    # Tally all data
    d_all = pd.DataFrame()
    trial = 1
    d_all = stack_data(info1['CONfns'], 'CON', trial)

    trial = d_all['trial'].max() + 1
    d_all = stack_data(info1['ADPfns'], 'ADP', trial)

    trial = d_all['trial'].max() + 1
    d_all = stack_data(info2['ADPfns'], 'ADP', trial)

    trial = d_all['trial'].max() + 1
    d_all = stack_data(info2['DADfns'], 'DAD', trial)

    temp = pd.DataFrame({'trial': d_all['trial'].unique()})
    temp['errblock'] = pd.qcut(temp['trial'], q=72, labels=False)
    temp['timeblock'] = pd.qcut(temp['trial'], q=12, labels=False)
    d_all = d_all.merge(temp, on='trial')

    temp = d_all.groupby(['trial', 'inflag']).cumcount() + 1
    temp = temp.groupby(['trial', 'inflag']).min().reset_index()
    temp_count = temp.groupby('trial').size().reset_index(name='n')
    temp_count = temp_count[temp_count['n'] > 1]
    temp_w = temp[temp['trial'].isin(temp_count['trial'])].pivot(index='trial', columns='inflag', values=0)

    d_all3 = d_all.merge(temp_w, on='trial')
    d_all3 = d_all3[d_all3['sample_id'] <= d_all3['inflag_min_0']]
    d_all3['xrl'] = d_all3.groupby('trial')['xr'].shift()
    d_all3['yrl'] = d_all3.groupby('trial')['yr'].shift()
    d_all3['xrf'] = d_all3.groupby('trial')['xr'].shift(-1)
    d_all3['yrf'] = d_all3.groupby('trial')['yr'].shift(-1)

    d_all4 = d_all3.copy()
    d_all4['diffangl'] = d_all4.apply(lambda row: comp_angle([row['xr'], row['yr']], [row['xrl'], row['yrl']]), axis=1)
    d_all4['diffangf'] = d_all4.apply(lambda row: comp_angle([row['xr'], row['yr']], [row['xrf'], row['yrf']]), axis=1)
    d_all4['distvl'] = np.linalg.norm(d_all4[['xr', 'yr']].values - d_all4[['xrl', 'yrl']].values, axis=1)
    d_all4['distvf'] = np.linalg.norm(d_all4[['xr', 'yr']].values - d_all4[['xrf', 'yrf']].values, axis=1)
    d_all4['absdiffangl'] = np.abs(d_all4['diffangl'])
    d_all4['absdiffangf'] = np.abs(d_all4['diffangf'])

    d_all4_summ = d_all4.groupby('trial').agg({'absdiffangl': lambda x: x.max(skipna=True), 
                                               'absdiffangf': lambda x: x.max(skipna=True)}).reset_index()

    d_all4_velocity_l = d_all4.loc[d_all4.groupby('trial')['distvl'].idxmax()]
    d_all4_velocity_f = d_all4.loc[d_all4.groupby('trial')['distvf'].idxmax()]

    good_curv_by_peak_l_vol = d_all4_velocity_l[d_all4_velocity_l['max_peak_l_end_angdiff'] <= 20][['trial', 'max_peak_l_end_angdiff']]
    good_curv_by_peak_f_vol = d_all4_velocity_f[d_all4_velocity_f['max_peak_f_end_angdiff'] <= 20][['trial', 'max_peak_f_end_angdiff']]

    good_curv_sample2end = d_all4_summ[d_all4_summ['max_absdiffang_l'] <= 90][['trial']]

    d_all2 = d_all4[d_all4['inflag'] == 1].copy()

    # Plotting and saving figures
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(d_all2['xr'], d_all2['yr'], color='blue', alpha=0.3, label='Trajectory')
    ax1.scatter(tarpos_cords['x'], tarpos_cords['y'], color='black', s=100, label='Target Positions')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    plt.title(sid)
    plt.savefig(f'figures/{sid}_trajectory.png')
    plt.close(f1)

    d_summ = d_all2.groupby(['trial', 'condition', 'errblock', 'timeblock']).agg({'tarpos': 'mean', 
                                                                                  'xrc': 'mean', 
                                                                                  'yrc': 'mean', 
                                                                                  't1': 'mean', 
                                                                                  't3': 'mean', 
                                                                                  't4': 'mean'}).reset_index()

    d_summ = d_summ.merge(tarpos_cords, left_on='tarpos', right_index=True)
    d_summ['angle'] = d_summ.apply(lambda row: comp_angle([row['xrc'], row['yrc']], [row['x'], row['y']]), axis=1)
    d_summ['rt'] = (d_summ['t3'] - d_summ['t1']) * 1000
    d_summ['mt'] = (d_summ['t4'] - d_summ['t3']) * 1000

    temp = d_summ.groupby('condition')['errblock'].agg(['min', 'max']).reset_index()
    print(temp)

    d_cross_bk = d_summ.groupby(['errblock', 'condition']).agg({'angle': 'median'}).reset_index()
    d_cross_bk.to_pickle(f'processed/{sid}_error.pkl')

    d_times_bk = d_summ.groupby(['timeblock', 'condition']).agg({'rt': 'median', 'mt': 'median'}).reset_index()
    d_times_bk.to_pickle(f'processed/{sid}_times.pkl')
