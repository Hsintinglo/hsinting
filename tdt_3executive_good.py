import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pyreadr
import statsmodels.api as sm
from scipy.stats import pearsonr

# 读取文件列表
path = 'D:/Exp/vat_group/plot_individuals/processed'
pattern = r'2024.*error\.RDS'
errfns = [os.path.join(path, fn) for fn in os.listdir(path) if re.match(pattern, fn)]

# 提取 sids
sids = [re.search(r'\d{8}_[A-Za-z]{2}', fn).group(0) for fn in errfns if re.search(r'\d{8}_[A-Za-z]{2}', fn)]
print("Extracted sids:", sids)

# 检查一个 .RDS 文件结构
try:
    sample_file = pyreadr.read_r(errfns[0])
    sample_file_df = list(sample_file.values())[0]
    print("Sample file structure:", sample_file_df.columns)
except Exception as e:
    print(f"Error reading sample file: {e}")

# 处理数据
def proc(sids_select, angle_column):
    errfns_select = [fn for fn in errfns if any(sid in fn for sid in sids_select)]
    dall = pd.DataFrame()
    
    for fn in errfns_select:
        result = pyreadr.read_r(fn)
        d = list(result.values())[0]
        dall = pd.concat([dall, d], ignore_index=True)
    
    # 检查是否存在指定的 angle 列
    if angle_column not in dall.columns:
        raise ValueError(f"Column '{angle_column}' not found in data")
    
    z_scores = np.abs((dall[angle_column] - dall[angle_column].mean()) / dall[angle_column].std())
    dall = dall[z_scores < 3]

    dall_summ = dall.groupby(['errblock', 'condition']).agg(
        error=(angle_column, 'mean'),
        sd=(angle_column, 'std'),
        n=(angle_column, 'size')
    ).reset_index()
    dall_summ['se'] = dall_summ['sd'] / np.sqrt(dall_summ['n'])
    dall_summ['upper'] = dall_summ['error'] + dall_summ['se']
    dall_summ['lower'] = dall_summ['error'] - dall_summ['se']

    return dall_summ

selected_sids = sids
dall_summ = proc(selected_sids, 'angle')

# TDT 数据
session = [1, 2, 3, 4, 5]
tdt_data = [
    #[170, 140, 129, 100, 60],
    #[180, 160, 135, 100, 90],
    #[210, 185, 150, 100, 55],
    #[130, 110, 90, 70, 55],
    [200, 175, 140, 130, 100],
    #[275, 225, 195, 150, 115],
    [295, 260, 235, 220, 200],
    [250, 237, 215, 195, 175],
    #[340, 338, 335, 295, 280],
    #[264, 233, 201, 170, 138],
    #[225, 190, 155, 125, 75],
    #[275, 255, 190, 175, 112],
    #[298, 248, 195, 175, 140],
    [275, 230, 198, 150, 130],
    #[300, 258, 202, 148, 75],
    #[152, 105, 60, 25, 20],
    #[480, 460, 440, 420, 400],
    #[218, 172, 148, 115, 60],
    #[272, 260, 240, 215, 198],
    #[478, 420, 378, 320, 280],
    [279, 253, 216, 177, 134]
    #[208, 165, 121, 86, 47]
]

# 计算每个受试者的 TDT slope
def calculate_slope(tdt_data, session):
    slopes = []
    X = np.array(session).reshape(-1, 1)
    for tdt in tdt_data:
        y = np.array(tdt)
        reg = LinearRegression().fit(X, y)
        slopes.append(reg.coef_[0])
    return slopes

tdt_slopes = calculate_slope(tdt_data, session)
vat_data = dall_summ['error'][:len(tdt_slopes)]

# 确保数据长度匹配
assert len(tdt_slopes) == len(vat_data), "TDT slopes and VAT data length mismatch."

# 各种执行任务表现 / 3 executive function median = 108.5
DCCS = [ 123, 121, 115, 132, 116] #median=99
#DCCS = [101, 100, 88, 95, 123, 85, 121, 115, 86, 98, 109, 93, 90, 132, 93, 94, 101, 121, 81, 104, 116, 89]
Flanker = [118, 108, 123, 133, 115] #median=109
#Flanker = [109, 106, 118, 104, 118, 113, 108, 123, 128, 111, 106, 109, 96, 133, 106, 102, 102, 125, 67, 136, 115, 101]
Visual_reasoning = [109, 128, 127, 119, 109] #median=108.5
#Visual_reasoning = [105, 108, 114, 106, 109, 103, 128, 127, 109, 106, 105, 109, 109, 119, 111, 105, 77, 101, 96, 124, 109, 85]

# 创建包含所有数据的 DataFrame
data = pd.DataFrame({
    'TDT slope': tdt_slopes,
    'DCCS': DCCS[:len(tdt_slopes)],
    'Flanker': Flanker[:len(tdt_slopes)],
    'Visual reasoning': Visual_reasoning[:len(tdt_slopes)]
})

# 执行多元线性回归
X = data[['DCCS', 'Flanker', 'Visual reasoning']]
X = sm.add_constant(X)
y = data['TDT slope']
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

# 绘制所有执行功能与 TDT 的散点图，并显示回归线和统计值
fig, ax = plt.subplots(figsize=(18, 9))

executive_functions = ['DCCS', 'Flanker', 'Visual reasoning']
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

# 初始化统计信息文本的位置
text_y_positions = [0.9, 0.75, 0.6]

for (ef, color, marker, text_y) in zip(executive_functions, colors, markers, text_y_positions):
    x = data[ef]
    y = data['TDT slope']
    
    ax.scatter(x, y, color=color, marker=marker, label=ef)
    
    # 计算相关性系数和 p 值
    corr_coef, p_value = pearsonr(x, y)
    
    # 线性回归
    X_single = sm.add_constant(x)
    model_single = sm.OLS(y, X_single).fit()
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred = model_single.predict(sm.add_constant(x_pred))
    
    ax.plot(x_pred, y_pred, color=color, linestyle='--')
    
    # 获取 R-squared
    r_squared = model_single.rsquared
    
    # 在图中添加统计值文本
    stats_text = f'{ef}\n$r^2$: {r_squared:.2f}\nCorr: {corr_coef:.2f}\np: {p_value:.3f}'
    fig.text(0.95, text_y, stats_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# 计算并显示多元线性回归的 R-squared 和 Adjusted R-squared
r_squared_multi = model.rsquared
adj_r_squared_multi = model.rsquared_adj

# 在右侧添加统计信息
stats_text_multi = f'Multiple Linear Regression\n$r^2$: {r_squared_multi:.2f}\nAdj $r^2$: {adj_r_squared_multi:.2f}'
fig.text(0.95, 0.45, stats_text_multi, transform=fig.transFigure, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlabel('Executive Functions')
ax.set_ylabel('TDT slope')
ax.set_title('Executive Functions vs. TDT slope')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.1))
plt.subplots_adjust(right=0.8)  # 调整右边距，给统计信息留出空间
plt.show()
