import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pyreadr
from scipy import stats

# 讀取檔案列表
path = 'D:/Exp/vat_group/plot_individuals/processed'
pattern = r'2024.*error\.RDS'
errfns = [os.path.join(path, fn) for fn in os.listdir(path) if re.match(pattern, fn)]

# 確認是否有匹配的文件
if not errfns:
    raise FileNotFoundError(f"No files matching the pattern {pattern} found in the directory {path}")

# 提取 sids
sids = []
for fn in errfns:
    print(f"Checking filename: {fn}")  # 打印文件名以供調試
    match = re.search(r'\d{8}_[A-Za-z]{2}', fn)
    if match:
        sids.append(match.group(0))
        print(f"Found sid: {match.group(0)}")  # 打印找到的 sid
    else:
        print("No valid sid found")  # 打印未找到 sid 的情況
print("Extracted sids:", sids)

# 確認是否有提取到的 sids
if not sids:
    raise ValueError("No valid sids found in the filenames")

# 檢查一個 .RDS 文件結構
sample_file = pyreadr.read_r(errfns[0])
sample_file_df = list(sample_file.values())[0]
print("Sample file structure:", sample_file_df.columns)

# 假設 angle 欄位名稱為 'angle'，如果不同請替換
angle_column = 'angle'  # 根據實際情況修改

# 處理資料
def proc(sids_select, angle_column):
    errfns_select = [fn for fn in errfns if any(sid in fn for sid in sids_select)]
    dall = pd.DataFrame()
    
    for fn in errfns_select:
        result = pyreadr.read_r(fn)
        d = list(result.values())[0]
        dall = pd.concat([dall, d], ignore_index=True)
    
    # 排除異常值 (此處簡單處理，實際應用中可能需要更複雜的方法)
    z_scores = np.abs((dall[angle_column] - dall[angle_column].mean()) / dall[angle_column].std())
    dall = dall[z_scores < 3]

    # 計算總結數據
    dall_summ = dall.groupby(['errblock', 'condition']).agg(
        error=(angle_column, 'mean'),
        sd=(angle_column, 'std'),
        n=(angle_column, 'size')
    ).reset_index()
    dall_summ['se'] = dall_summ['sd'] / np.sqrt(dall_summ['n'])
    dall_summ['upper'] = dall_summ['error'] + dall_summ['se']
    dall_summ['lower'] = dall_summ['error'] - dall_summ['se']

    return dall_summ

# 選擇所有的 sids
selected_sids = sids

# 處理數據
dall_summ = proc(selected_sids, angle_column)

# TDT 數據
session = [1, 2, 3, 4, 5]
tdt_data = [
    [170, 140, 129, 100, 60],
    [180, 160, 135, 100, 90],
    [210, 185, 150, 100, 55],
    [130, 110, 90, 70, 55],
    [200, 175, 140, 130, 100],
    [275, 225, 195, 150, 115],
    [295, 260, 235, 220, 200],
    [250, 237, 215, 195, 175],
    [340, 338, 335, 295, 280],
    [264, 233, 201, 170, 138],
    [225, 190, 155, 125, 75],
    [275, 255, 190, 175, 112],
    [298, 248, 195, 175, 140],
    [275, 230, 198, 150, 130],
    [300, 258, 202, 148, 75],
    [152, 105, 60, 25, 20],
    [480, 460, 440, 420, 400],
    [218, 172, 148, 115, 60],
    [272, 260, 240, 215, 198],
    [478, 420, 378, 320, 280],
    [279, 253, 216, 177, 134],
    [208, 165, 121, 86, 47],
    [336, 304, 295, 278, 260]
]

# 計算每個受試者的 TDT slope
def calculate_slope(tdt_data, session):
    slopes = []
    X = np.array(session).reshape(-1, 1)
    for tdt in tdt_data:
        y = np.array(tdt)
        reg = LinearRegression().fit(X, y)
        slopes.append(reg.coef_[0])
    return slopes

tdt_slopes = calculate_slope(tdt_data, session)

# 創建包含 TDT slope 和 VAT 數據的 DataFrame，確保數據長度匹配
num_slopes = len(tdt_slopes)
vat_data = dall_summ['error'][:num_slopes]  # 使用 error 作為 VAT 數據

# 確保 tdt_slopes 和 vat 數據的長度匹配
assert len(tdt_slopes) == len(vat_data), "TDT slopes and VAT data length mismatch."

# 創建包含 TDT slope 和 VAT 數據的 DataFrame
data = pd.DataFrame({
    'TDT slope': tdt_slopes,
    'VAT': vat_data
})

# 執行線性回歸並計算 p 值
X = data[['TDT slope']]
y = data['VAT']
reg = LinearRegression().fit(X, y)

# 使用 scipy.stats.linregress 計算 p 值
slope, intercept, r_value, p_value, std_err = stats.linregress(data['TDT slope'], data['VAT'])

# 繪製散點圖和回歸線
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TDT slope', y='VAT', data=data, color='blue', label='Data points')
plt.plot(data['TDT slope'], reg.predict(X), color='red', linewidth=2, label='Regression line')
plt.xlabel('TDT slope')
plt.ylabel('VAT')
plt.title('Linear Regression of TDT vs VAT slope')
plt.legend()

# 添加回歸參數到圖中，不重疊
plt.text(0.05, 0.95, f'Intercept: {intercept:.2f}', fontsize=10, color='red', ha='left', va='top', transform=plt.gca().transAxes)
plt.text(0.05, 0.90, f'Coefficient: {slope:.2f}', fontsize=10, color='red', ha='left', va='top', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'R^2: {r_value**2:.2f}', fontsize=10, color='red', ha='left', va='top', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'p-value: {p_value:.4f}', fontsize=10, color='red', ha='left', va='top', transform=plt.gca().transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 調整佈局以便給文字留出空間
plt.show()
