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
DCCS = [101, 100, 88, 95, 123, 85, 121, 115, 86, 98, 109, 93, 90, 132, 93, 94, 101, 121, 81, 104, 116, 89, 98] #, 102, 137, 99
Flanker = [109, 106, 118, 104, 118, 113, 108, 123, 128, 111, 106, 109, 96, 133, 106, 102, 102, 125, 67, 136, 115, 101, 97] #, 102, 117, 113
Visual_reasoning = [105, 108, 114, 106, 109, 103, 128, 127, 109, 106, 105, 109, 109, 119, 111, 105, 77, 101, 96, 124, 109, 85, 111] #, 134, 115, 106

# 創建包含所有數據的 DataFrame
data = pd.DataFrame({
    'TDT slope': tdt_slopes,
    'VAT': vat_data,
    'DCCS': DCCS,
    'Flanker': Flanker,
    'Visual reasoning': Visual_reasoning
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
mediators = ['DCCS', 'Flanker', 'Visual reasoning']

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

# 繪製每對變量的散點圖，並在圖上顯示 p 值、回歸係數和 R 平方值
def annotate_regression(ax, x, y):
    X = sm.add_constant(data[x])
    model = sm.OLS(data[y], X).fit()
    coef = model.params[1]
    r_squared = model.rsquared
    p_value = model.pvalues[1]
    annotation = f'$R^2={r_squared:.2f}$\n$p={p_value:.3f}$\n$coef={coef:.2f}$'
    ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# 使用 seaborn 的 pairplot 並添加回歸線和統計信息
g = sns.pairplot(data, kind='reg', height=2.5)
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    annotate_regression(g.axes[i, j], data.columns[j], data.columns[i])

plt.suptitle("Scatterplot Matrix of All Variables with Linear Regression Lines", y=1.02)
plt.savefig('small_plots.png')
plt.show()

# 繪製大圖
def plot_large_scatterplots(data):
    sns.set(style='whitegrid')
    for x in data.columns:
        for y in data.columns:
            if x != y:
                plt.figure(figsize=(10, 8))
                ax = sns.regplot(x=x, y=y, data=data)
                annotate_regression(ax, x, y)
                plt.title(f'Scatterplot of {x} vs {y}')
                plt.savefig(f'large_plot_{x}_vs_{y}.png')
                plt.close()

plot_large_scatterplots(data)
