import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pyreadr
import statsmodels.api as sm
from scipy.stats import norm

# 檢查並安裝缺少的套件
def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

packages = ['os', 're', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'pyreadr', 'statsmodels', 'scipy']
for package in packages:
    install_and_import(package)

# 讀取檔案列表
path = 'D:/Exp/vat_group/plot_individuals/processed'
pattern = r'2024.*error\.RDS'
errfns = [os.path.join(path, fn) for fn in os.listdir(path) if re.match(pattern, fn)]

# 提取 sids
sids = [re.search(r'\d{8}_[A-Za-z]{2}', fn).group(0) for fn in errfns if re.search(r'\d{8}_[A-Za-z]{2}', fn)]
print("Extracted sids:", sids)

# 檢查一個 .RDS 文件結構
try:
    sample_file = pyreadr.read_r(errfns[0])
    sample_file_df = list(sample_file.values())[0]
    print("Sample file structure:", sample_file_df.columns)
except Exception as e:
    print(f"Error reading sample file: {e}")

# 處理資料
def proc(sids_select, angle_column):
    errfns_select = [fn for fn in errfns if any(sid in fn for sid in sids_select)]
    dall = pd.DataFrame()
    
    for fn in errfns_select:
        result = pyreadr.read_r(fn)
        d = list(result.values())[0]
        dall = pd.concat([dall, d], ignore_index=True)
    
    # 檢查是否存在指定的 angle 欄位
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
vat_data = dall_summ['error'][:len(tdt_slopes)]

# 確保數據長度匹配
assert len(tdt_slopes) == len(vat_data), "TDT slopes and VAT data length mismatch."

# 各種執行任務表現
DCCS = [101, 100, 88, 95, 123, 85, 121, 115, 86, 98, 109, 93, 90, 132, 93, 94, 101, 121, 81, 104, 116, 89, 98, 102, 137, 99]
Flanker = [109, 106, 118, 104, 118, 113, 108, 123, 128, 111, 106, 109, 96, 133, 106, 102, 102, 125, 67, 136, 115, 101, 97, 102, 117, 113]
Visual_reasing = [105, 108, 114, 106, 109, 103, 128, 127, 109, 106, 105, 109, 109, 119, 111, 105, 77, 101, 96, 124, 109, 85, 111, 134, 115, 106]
Dexterity_dominant = [109, 131, 103, 103, 125, 65, 109, 81, 109, 136, 98, 114, 109, 114, 98, 109, 104, 87, 87, 71, 120, 93, 98, 103, 103, 93]
Dexterity_non_dominant = [110, 122, 91, 91, 85, 72, 129, 91, 98, 129, 122, 104, 91, 110, 110, 110, 92, 92, 85, 97, 104, 86, 98, 91, 104, 104]

# 創建包含所有數據的 DataFrame
data = pd.DataFrame({
    'TDT slope': tdt_slopes,
    'VAT': vat_data,
    'DCCS': DCCS,
    'Flanker': Flanker,
    'Visual reasoning': Visual_reasing,
    'Dexterity dominant': Dexterity_dominant,
    'Dexterity non-dominant': Dexterity_non_dominant
})

# Sobel 測試
def sobel_test(a, b, sa, sb):
    sobel_stat = a * b / np.sqrt(b**2 * sa**2 + a**2 * sb**2)
    p_value = 2 * (1 - norm.cdf(np.abs(sobel_stat)))
    return sobel_stat, p_value

# 執行中介分析
def manual_mediation_analysis(data, mediator_var, iv, dv):
    # 中介變量對自變量的回歸
    mediator_model = sm.OLS(data[mediator_var], sm.add_constant(data[iv])).fit()
    # 結果變量對自變量和中介變量的回歸
    outcome_model = sm.OLS(data[dv], sm.add_constant(data[[iv, mediator_var]])).fit()
    
    a = mediator_model.params[iv]
    b = outcome_model.params[mediator_var]
    sa = mediator_model.bse[iv]
    sb = outcome_model.bse[mediator_var]
    
    sobel_stat, p_value = sobel_test(a, b, sa, sb)
    
    return {
        'mediator_model_summary': mediator_model.summary(),
        'outcome_model_summary': outcome_model.summary(),
        'sobel_stat': sobel_stat,
        'p_value': p_value
    }

# 中介變量的列表
mediators = ['DCCS', 'Flanker', 'Visual reasoning', 'Dexterity dominant', 'Dexterity non-dominant']

# 執行每個中介變量的分析
mediation_results = {}
for mediator in mediators:
    try:
        result = manual_mediation_analysis(data, mediator, 'TDT slope', 'VAT')
        mediation_results[mediator] = result
    except Exception as e:
        print(f"Error in mediation analysis for {mediator}: {e}")

# 輸出結果
for mediator, result in mediation_results.items():
    print(f"Mediation analysis for {mediator}:")
    print(result['mediator_model_summary'])
    print(result['outcome_model_summary'])
    print(f"Sobel test statistic: {result['sobel_stat']}, p-value: {result['p_value']}")
    print("\n")

# 繪製回歸模型的圖像
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# 繪製 TDT slope 與 VAT 的回歸圖
X = np.array(data['TDT slope']).reshape(-1, 1)
y = data['VAT']
reg = LinearRegression().fit(X, y)

axs[0, 0].scatter(data['TDT slope'], data['VAT'], color='blue', label='Data points')
axs[0, 0].plot(data['TDT slope'], reg.predict(X), color='red', linewidth=2, label='Regression line')
axs[0, 0].set_xlabel('TDT slope')
axs[0, 0].set_ylabel('VAT')
axs[0, 0].set_title('Linear Regression of TDT slope vs VAT')
axs[0, 0].legend()

# 繪製每個中介變量的回歸圖
for i, mediator in enumerate(mediators):
    row, col = (i + 1) // 2, (i + 1) % 2

    # TDT slope 與中介變量的回歸圖
    axs[row, col].scatter(data['TDT slope'], data[mediator], color='blue', label=f'{mediator} vs TDT slope')
    axs[row, col].plot(data['TDT slope'], sm.OLS(data[mediator], sm.add_constant(data['TDT slope'])).fit().predict(sm.add_constant(data['TDT slope'])), color='red', linewidth=2, label='Regression line')
    axs[row, col].set_xlabel('TDT slope')
    axs[row, col].set_ylabel(mediator)
    axs[row, col].set_title(f'Regression of TDT slope vs {mediator}')
    axs[row, col].legend()

plt.tight_layout()
plt.show()

# 繪製每個中介變量與 VAT 的回歸圖
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

for i, mediator in enumerate(mediators):
    row, col = i // 2, i % 2

    axs[row, col].scatter(data[mediator], data['VAT'], color='blue', label=f'{mediator} vs VAT')
    axs[row, col].plot(data[mediator], sm.OLS(data['VAT'], sm.add_constant(data[mediator])).fit().predict(sm.add_constant(data[mediator])), color='red', linewidth=2, label='Regression line')
    axs[row, col].set_xlabel(mediator)
    axs[row, col].set_ylabel('VAT')
    axs[row, col].set_title(f'Regression of {mediator} vs VAT')
    axs[row, col].legend()

plt.tight_layout()
plt.show()
