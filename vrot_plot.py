import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# 创建数据框
data = d_cross_bk()

# 设置 x 轴和 y 轴的标签
xlabstr_time = 'Blocks (24-trials)'
xlabstr_err = 'Mini-blocks (4-trials)'
ylabstr = "angular error (deg)"

# 设置绘图风格
sns.set(style="whitegrid")

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制散点图
sns.scatterplot(x='errblock', y='angle', hue='condition', data=data)

# 线性拟合
lm1 = smf.ols(formula='angle ~ errblock', data=data).fit()
lm2 = smf.ols(formula='angle ~ np.log(errblock)', data=data).fit()

# 绘制拟合曲线
x_vals = np.linspace(data['errblock'].min(), data['errblock'].max(), 100)
y_vals1 = lm1.params.Intercept + lm1.params.errblock * x_vals
y_vals2 = lm2.params.Intercept + lm2.params['np.log(errblock)'] * np.log(x_vals)

plt.plot(x_vals, y_vals1, linestyle='-', color='blue', label='Linear Fit')
plt.plot(x_vals, y_vals2, linestyle='--', color='black', label='Log Fit')

# 设置图表标题和标签
plt.title(sid)
plt.xlabel(xlabstr_err)
plt.ylabel(ylabstr)

# 显示图例
plt.legend(loc='lower right')

# 保存图表
plt.savefig(f'D:/visuomotor task/visuomotor2024-master/vrot_scripts/figures/error/{sid}_error.png', dpi=300)

# 显示图表
plt.show()
