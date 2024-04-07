import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def getfns(droot, sids):
    result = {}
    for sid in sids:
        subpaths = [os.path.join(droot, d) for d in os.listdir(droot) if os.path.isdir(os.path.join(droot, d))]
        dat_dir = [d for d in subpaths if os.path.basename(d) == sid]
        if not dat_dir:
            continue
        dat_dir = dat_dir[0]
        CONfns = [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.startswith('CON_') and f.endswith('.npy')]
        ADPfns = [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.startswith('ADP_') and f.endswith('.npy')]
        DADfns = [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.startswith('DAD_') and f.endswith('.npy')]
        result[sid] = {'CONfns': CONfns, 'ADPfns': ADPfns, 'DADfns': DADfns}
    return result

def stack_data(fns, condstr):
    d_all = pd.DataFrame()
    trial = 1
    for fn in fns:
        d = np.load(fn)
        d = pd.DataFrame(d, columns=['x', 'y', 'inflag', 'xr', 'yr', 'tarpos', 'xrc', 'yrc', 't1', 't3', 't4'])
        d['trial'] = trial
        d['condition'] = condstr
        trial += 1
        d_all = pd.concat([d_all, d])
    return d_all

# Replace ntile function with pandas' quantile function
def create_blocks(trial_col, n_blocks):
    return pd.qcut(trial_col, n_blocks, labels=False) + 1

# Replace left_join with merge function
def left_join(df1, df2, on):
    return df1.merge(df2, on=on, how='left')

# Define comp_angle function
def comp_angle(u, v):
    if np.isnan(u).any() or np.isnan(v).any():
        return np.nan
    theta = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
    theta = np.degrees(theta)
    if theta > 180:
        theta = 360 - theta
    return theta

# Replace rowwise function with iterating over DataFrame rows
def compute_diff_angles(d_all3):
    diff_angles = []
    for _, row in d_all3.iterrows():
        xr, yr, xrl, yrl, xrf, yrf = row['xr'], row['yr'], row['xrl'], row['yrl'], row['xrf'], row['yrf']
        diffangl = comp_angle([xr, yr], [xrl, yrl])
        diffangf = comp_angle([xr, yr], [xrf, yrf])
        distvl = np.linalg.norm(np.array([xr, yr]) - np.array([xrl, yrl]))
        distvf = np.linalg.norm(np.array([xr, yr]) - np.array([xrf, yrf]))
        absdiffangl = abs(diffangl)
        absdiffangf = abs(diffangf)
        diff_angles.append([diffangl, diffangf, distvl, distvf, absdiffangl, absdiffangf])
    return pd.DataFrame(diff_angles, columns=['diffangl', 'diffangf', 'distvl', 'distvf', 'absdiffangl', 'absdiffangf'])

# Replace group_by and slice functions with groupby and idxmax functions
def find_max_distvl(d_all4):
    return d_all4.groupby('trial').apply(lambda x: x.loc[x['distvl'].idxmax()])

def find_max_distvf(d_all4):
    return d_all4.groupby('trial').apply(lambda x: x.loc[x['distvf'].idxmax()])

# Replace filter function with boolean indexing
def filter_good_curv_by_peak(d_all4_velocity, threshold):
    return d_all4_velocity[d_all4_velocity['max_peak_end_angdiff'] <= threshold]

def filter_good_curv_sample2end(d_all4_summ, threshold):
    return d_all4_summ[d_all4_summ['max_absdiffang_l'] <= threshold]

# Replace dplyr::filter with boolean indexing
def filter_inflag_eq_1(d_all4):
    return d_all4[d_all4['inflag'] == 1]

# Define tarpos_cords
tarpos_cords = pd.DataFrame({'x': [-0.4, 0.4, 0, 0], 'y': [0, 0, 0.4, -0.4]})

# Replace geom_line, geom_point, and geom_point functions with matplotlib plot functions
def plot_trace(d_all2):
    plt.figure()
    for name, group in d_all2.groupby('trial'):
        plt.plot(group['xr'], group['yr'], alpha=0.3)
        plt.scatter(group['xr'], group['yr'])
    plt.scatter(tarpos_cords['x'], tarpos_cords['y'], color='black', s=100)
    plt.xlabel('xr')
    plt.ylabel('yr')
    plt.legend()
    plt.show()

# Replace group_by and summarize functions with groupby and aggregate functions
def summarize_data(d_summ):
    return d_summ.groupby(['trial', 'condition', 'errblock', 'timeblock']).agg(
        tarpos=('tarpos', 'mean'),
        xrc=('xrc', 'mean'),
        yrc=('yrc', 'mean'),
        t1=('t1', 'mean'),
        t3=('t3', 'mean'),
        t4=('t4', 'mean')
    )

# Define saveRDS function (not available in Python, need to use other methods to save data)
def saveRDS(data, file):
    pass  # Placeholder for saving data in RDS format

# Define sid variable
sids = ['20240327_TJ', '20240327_PS', '20240329_MT', '20240402_YS', '20240403_TT', '20240403_MJ', '20240403_SX']

# Define droot1 and droot2 variables
droot1 = C:/Users/User/桌面/vrot analysis/visuomotor2024-master/vrot_first/data #'../vrot_first/data'
droot2 = C:/Users/User/桌面/vrot analysis/visuomotor2024-master/vrot_second/data #'../vrot_second/data'

# Use getfns function to get file information
info1 = getfns(droot1, sids)
info2 = getfns(droot2, sids)

# Initialize d_all DataFrame
d_all = pd.DataFrame()

# Stack data for each condition
trial = 1
for sid in sids:
    d_all = pd.concat([d_all, stack_data(info1[sid]['CONfns'], 'CON')])
    trial = max(d_all['trial']) + 1
    d_all = pd.concat([d_all, stack_data(info1[sid]['ADPfns'], 'ADP')])
    trial = max(d_all['trial']) + 1
    d_all = pd.concat([d_all, stack_data(info2[sid]['ADPfns'], 'ADP')])
    trial = max(d_all['trial']) + 1
    d_all = pd.concat([d_all, stack_data(info2[sid]['DADfns'], 'DAD')])

# Create temp DataFrame
temp = pd.DataFrame({'trial': d_all['trial'].unique()})
temp['errblock'] = create_blocks(temp['trial'], 72)
temp['timeblock'] = create_blocks(temp['trial'], 12)

# Left join temp DataFrame with d_all DataFrame
d_all = left_join(d_all, temp, on='trial')

# Select required columns from d_all DataFrame and summarize data
temp = d_all[['trial', 'inflag']].groupby('trial').apply(lambda x: x.reset_index(drop=True)).reset_index()
temp['sample_id'] = temp.groupby(['trial', 'inflag']).cumcount() + 1
temp = temp.groupby(['trial', 'inflag']).agg(min_seg=('sample_id', 'min')).reset_index()
temp_count = temp['trial'].value_counts().reset_index()
temp_count.columns = ['trial', 'n']
temp_count = temp_count[temp_count['n'] > 1]
temp_w = temp[temp['trial'].isin(temp_count['trial'])].pivot(index='trial', columns='inflag', values='min_seg').add_prefix('inflag_min_').reset_index()

# Filter d_all DataFrame
d_all3 = d_all[d_all['trial'].isin(temp_count['trial'])].copy()
d_all3['sample_id'] = d_all3.groupby('trial').cumcount() + 1
d_all3 = left_join(d_all3, temp_w, on='trial')
d_all3 = d_all3[d_all3['sample_id'] <= d_all3['inflag_min_0']]
d_all3['xrl'] = d_all3.groupby('trial')['xr'].shift(1)
d_all3['yrl'] = d_all3.groupby('trial')['yr'].shift(1)
d_all3['xrf'] = d_all3.groupby('trial')['xr'].shift(-1)
d_all3['yrf'] = d_all3.groupby('trial')['yr'].shift(-1)

# Compute angular differences
d_all4 = compute_diff_angles(d_all3)

# Find maximum distvl and distvf
d_all4_velocity_l = find_max_distvl(d_all4)
d_all4_velocity_f = find_max_distvf(d_all4)

# Filter data based on conditions
good_curv_by_peak_l_vol = filter_good_curv_by_peak(d_all4_velocity_l, 20)
good_curv_by_peak_f_vol = filter_good_curv_by_peak(d_all4_velocity_f, 20)
good_curv_sample2end = filter_good_curv_sample2end(d_all4_summ, 90)

# Filter data by inflag value
d_all2 = filter_inflag_eq_1(d_all4)

# Plot trace
plot_trace(d_all2)

# Summarize data
d_summ = summarize_data(d_all2)

# Transform data and compute angle between target position and cross point
d_summ2 = d_summ.copy()
d_summ2['xrt'] = d_summ2['tarpos'].apply(lambda x: tarpos_cords['x'][x + 1])
d_summ2['yrt'] = d_summ2['tarpos'].apply(lambda x: tarpos_cords['y'][x + 1])
d_cross = d_summ2.apply(lambda row: comp_angle([row['xrc'], row['yrc']], [row['xrt'], row['yrt']]), axis=1)
d_cross = pd.DataFrame({'angle': d_cross, 'rt': (d_summ2['t3'] - d_summ2['t1']) * 1000, 'mt': (d_summ2['t4'] - d_summ2['t3']) * 1000})

# Save data
d_cross_bk = d_cross.groupby(['errblock', 'condition']).agg({'angle': 'median'}).reset_index()
d_times_bk = d_cross.groupby(['timeblock', 'condition']).agg({'rt': 'median', 'mt': 'median'}).reset_index()

# Save data (This part needs to be modified based on how you want to save the data in Python)
d_cross_bk.to_csv(f'processed/{sid}_error.csv', index=False)
d_times_bk.to_csv(f'processed/{sid}_times.csv', index=False)
