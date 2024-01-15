import seaborn as sns
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# violinplot.pdf
df = pd.read_csv('data_prepare/output/NSCLC1_main_results.csv', index_col = 0)

target_column = 'dfr'
random_values = np.random.uniform(-0.1, 0.1, size=len(df))
df[target_column] += random_values

sns.violinplot(x='group', y='dfr', data=df, inner=None)
sns.stripplot(x='group', y='dfr', data=df, jitter=True, color='black', alpha=1)

plt.title('Pirate Plot')
plt.xlabel('Label')
plt.ylabel('dfr')
plt.ylim(-0.2, 1.2)
fig = plt.gcf()
fig.set_size_inches(10, 10)

plt.savefig('figures/violinplot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

contingency_table = pd.crosstab(df['dfr'], df['group'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('p_value:', p_value)


def draw_pic(data_0, data_1, columns, x_lim_start, x_lim_end, output_file):

    values_0 = data_0[columns].values.flatten()
    values_1 = data_1[columns].values.flatten()

    bar_width = 0.35
    index = np.arange(len(columns))

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_0 = ax.barh(index, values_0, bar_width, label='Dataset1-600')
    bar_1 = ax.barh(index + bar_width, values_1, bar_width, label='Dataset1-600')

    for rect in bar_0:
        width = rect.get_width()
        ax.text(width + 0.001, rect.get_y() + rect.get_height() / 2, f'{width:.5f}', ha='left', va='center')

    for rect in bar_1:
        width = rect.get_width()
        ax.text(width + 0.001, rect.get_y() + rect.get_height() / 2, f'{width:.5f}', ha='left', va='center')

    ax.set_xlabel('Values')
    ax.set_ylabel('Metrics')
    ax.set_title('Comparison of Values')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(columns)
    ax.legend()

    ax.set_xlim(x_lim_start, x_lim_end)

    plt.tight_layout()

    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


# compare.csv
# df = pd.read_csv('figures/compare.csv', index_col=0)
df = pd.read_csv('figures/NSCLC1-compare.csv', index_col=0)
columns = ['hl_loss', 'log_cosh_loss', 'mae_loss', 'mse_loss', 'huber_loss']

# mse
data_0 = df[(df['indicator'] == 'mse') & (df['dataset'] == 0)]
data_1 = df[(df['indicator'] == 'mse') & (df['dataset'] == 1)]
x_lim_start = 0
x_lim_end = 0.004
mse_output_file = 'figures/mse2.pdf'
draw_pic(data_0, data_1, columns, x_lim_start, x_lim_end, mse_output_file)

# hl_value
data_0 = df[(df['indicator'] == 'HL_value') & (df['dataset'] == 0)]
data_1 = df[(df['indicator'] == 'HL_value') & (df['dataset'] == 1)]
x_lim_start = 0
x_lim_end = 1.8
hl_output_file = 'figures/hl_value.pdf'
draw_pic(data_0, data_1, columns, x_lim_start, x_lim_end, hl_output_file)

# p value
data_0 = df[(df['indicator'] == 'p_val') & (df['dataset'] == 0)]
data_1 = df[(df['indicator'] == 'p_val') & (df['dataset'] == 1)]
x_lim_start = 0.95
x_lim_end = 1.05
p_output_file = 'figures/p_value.pdf'
draw_pic(data_0, data_1, columns, x_lim_start, x_lim_end, p_output_file)