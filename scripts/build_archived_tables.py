# 论文实证分析：Fama-French五因子模型在中国A股市场的适用性检验
# 样本期间：2006年1月-2024年12月

# 导入必要的库
import os
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_1samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_ROOT = Path(os.environ.get("SAS_DATA_ROOT", "data"))
OUTPUT_DIR = Path(os.environ.get("FF5_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#%% 1. 数据读取与预处理
print("正在读取个股月度数据...")
stock_data = (
    pd.read_csv(DATA_ROOT / "个股月度交易数据" / "TRD_Mnth.csv", dtype={'Stkcd': str})
    .assign(
        ym=lambda x: x['Trdmnt'],
        stkcd=lambda x: x['Stkcd'].astype(str).str.zfill(6),
        return_monthly=lambda x: x['Mretwd'],
        size_monthly=lambda x: x['Msmvosd']
    )
    .loc[lambda x: (x['stkcd'].str[0].isin(['0','3','6'])) & 
                 (~x['stkcd'].str.startswith(('688','689')))]
    [['stkcd', 'ym', 'return_monthly', 'size_monthly']]
    .dropna()
    .query("'2006-01' <= ym <= '2024-12'")
)
print(f"个股数据读取完成，共{len(stock_data)}条记录")

# 读取无风险利率数据
print("正在读取无风险利率数据...")
rf_data = (
    pd.read_csv(DATA_ROOT / "无风险利率数据" / "TRD_Nrrate.csv")
    .assign(
        ym=lambda x: pd.to_datetime(x['Clsdt']).dt.strftime('%Y-%m'),
        rf_monthly=lambda x: x['Nrrmtdt'] / 100
    )
    [['ym', 'rf_monthly']]
    .drop_duplicates('ym')
    .query("'2006-01' <= ym <= '2024-12'")
)
print(f"无风险利率数据读取完成，共{len(rf_data)}条记录")

# 读取财务报表数据
print("正在读取财务报表数据...")
financial_data = (
    pd.read_csv(DATA_ROOT / "常用变量查询（年度）155452706" / "常用变量查询（年度）.csv", 
               encoding='utf-8', dtype={'Stkcd': str})
    .assign(
        stkcd=lambda x: x['Stkcd'].astype(str).str.zfill(6),
        ym=lambda x: x['accper'].astype(str).str[:4] + '-12'
    )
    .sort_values(['stkcd', 'ym'])
)
print(f"财务报表数据读取完成，共{len(financial_data)}条记录")

# 计算财务指标
print("正在计算财务指标...")
financial_data['lag_A003000000'] = financial_data.groupby('stkcd')['A003000000'].shift(1)
financial_data['investment'] = np.where(
    (financial_data['lag_A003000000'] > 0) & (financial_data['lag_A003000000'].notna()),
    (financial_data['A003000000'] - financial_data['lag_A003000000']) / financial_data['lag_A003000000'],
    np.nan
)
financial_data['bm'] = financial_data['A001000000a'] / financial_data['A003000000']
financial_data['roe'] = financial_data['B001300000'] / financial_data['A001000000a']

financial_indicators = (
    financial_data
    [['stkcd', 'ym', 'bm', 'roe', 'investment']]
    .dropna()
    .sort_values(['stkcd', 'ym'])
    .reset_index(drop=True)
)
print(f"财务指标计算完成，共{len(financial_indicators)}条记录")

#%% 2. 构建投资组合
def create_simple_annual_portfolios(stock_df, financial_df, rf_df):
    """简化的年度分组函数"""
    all_results = []
    
    for year in range(2006, 2025):
        grouping_date = f"{year}-12"
        grouping_stocks = stock_df[stock_df['ym'] == grouping_date].copy()
        
        if len(grouping_stocks) < 100:
            continue
        
        financial_year = year - 1
        financial_ym = f"{financial_year}-12"
        
        grouping_data = grouping_stocks.merge(
            financial_df[financial_df['ym'] == financial_ym],
            on='stkcd', how='inner'
        )
        
        if len(grouping_data) < 50:
            continue
        
        try:
            # 按市值分5组
            grouping_data['size_quintile'] = pd.qcut(
                grouping_data['size_monthly'], q=5, labels=['Small', '2', '3', '4', 'Big']
            ).astype(str)
            
            # 按账面市值比分5组
            grouping_data['bm_quintile'] = pd.qcut(
                grouping_data['bm'], q=5, labels=['Low', '2', '3', '4', 'High']
            ).astype(str)
            
            grouping_data['portfolio'] = grouping_data['size_quintile'] + '_' + grouping_data['bm_quintile']
            
            # 持有期：下一年1-12月
            if year < 2024:
                holding_months = [f"{year+1}-{month:02d}" for month in range(1, 13)]
            else:
                holding_months = []
            
            if holding_months:
                holding_data = stock_df[stock_df['ym'].isin(holding_months)]
                
                for portfolio in grouping_data['portfolio'].unique():
                    stocks = grouping_data[grouping_data['portfolio'] == portfolio]['stkcd']
                    returns = holding_data[holding_data['stkcd'].isin(stocks)]
                    
                    if len(returns) > 0:
                        monthly_avg = returns.groupby('ym')['return_monthly'].mean()
                        
                        for ym, ret in monthly_avg.items():
                            all_results.append({
                                'ym': ym, 'portfolio': portfolio, 'return_monthly': ret
                            })
        
        except Exception as e:
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(all_results)
    result_df = result_df.merge(rf_df, on='ym', how='inner')
    result_df['excess_return'] = result_df['return_monthly'] - result_df['rf_monthly']
    
    return result_df

# 应用年度分组
print("正在构建年度投资组合...")
annual_portfolio_data = create_simple_annual_portfolios(stock_data, financial_indicators, rf_data)
print(f"投资组合构建完成，共{len(annual_portfolio_data)}条记录，{annual_portfolio_data['portfolio'].nunique()}个组合")

#%% 3. 表1：基于市值与账面市值比的分组收益率
print("正在计算表1：投资组合收益率...")
portfolio_stats = annual_portfolio_data.groupby('portfolio').agg({
    'excess_return': ['mean', 'std', 'count']
}).round(6)
portfolio_stats.columns = ['mean_return', 'std_return', 'obs_count']
portfolio_stats = portfolio_stats.reset_index()

# 计算t统计量和p值
t_stats_results = []
for portfolio in portfolio_stats['portfolio'].unique():
    port_data = annual_portfolio_data[annual_portfolio_data['portfolio'] == portfolio]['excess_return']
    if len(port_data) > 1:
        t_stat, p_value = ttest_1samp(port_data, 0)
        t_stats_results.append({'portfolio': portfolio, 't_stat': t_stat, 'p_value': p_value})

t_stats_df = pd.DataFrame(t_stats_results)
portfolio_stats = portfolio_stats.merge(t_stats_df, on='portfolio', how='left')

def add_stars(p):
    """添加显著性星号"""
    if pd.isna(p): return ""
    if p <= 0.01: return "***"
    elif p <= 0.05: return "**"
    elif p <= 0.1: return "*"
    else: return ""

portfolio_stats['stars'] = portfolio_stats['p_value'].apply(add_stars)
portfolio_stats[['size_group', 'bm_group']] = portfolio_stats['portfolio'].str.split('_', expand=True)

# 创建透视表
size_order = ['Small', '2', '3', '4', 'Big']
bm_order = ['Low', '2', '3', '4', 'High']

mean_return_table = portfolio_stats.pivot_table(
    index='size_group', columns='bm_group', values='mean_return', aggfunc='mean'
).reindex(index=size_order, columns=bm_order)

# 保存表1数据
table1_results = []
for size_group in size_order:
    for bm_group in bm_order:
        portfolio = f"{size_group}_{bm_group}"
        portfolio_data = portfolio_stats[portfolio_stats['portfolio'] == portfolio]
        if not portfolio_data.empty:
            row_data = portfolio_data.iloc[0]
            table1_results.append({
                '市值分组': size_group, '账面市值比分组': bm_group, '组合标识': portfolio,
                '平均超额收益率(%)': row_data['mean_return'] * 100,
                '收益率标准差(%)': row_data['std_return'] * 100,
                '观测数': row_data['obs_count'], 't统计量': row_data['t_stat'],
                'p值': row_data['p_value'], '显著性': row_data['stars']
            })

table1_df = pd.DataFrame(table1_results)
table1_df.to_excel(OUTPUT_DIR / "论文表1_基于市值与账面市值比的分组收益率.xlsx", index=False)
print("表1已保存")

#%% 4. 表2：五因子描述性统计
print("正在读取五因子数据...")
ff5_data = pd.read_csv(DATA_ROOT / "ff月度五因子数据" / "STK_MKT_FIVEFACMONTH.csv")

ff5_clean = (
    ff5_data
    .query("MarkettypeID == 'P9709' & Portfolios == 1")
    .assign(
        ym=lambda x: x['TradingMonth'],
        MKT=lambda x: x['RiskPremium1'], SMB=lambda x: x['SMB1'],
        HML=lambda x: x['HML1'], RMW=lambda x: x['RMW1'], CMA=lambda x: x['CMA1']
    )
    [['ym', 'MKT', 'SMB', 'HML', 'RMW', 'CMA']]
    .dropna().query("'2006-01' <= ym <= '2024-12'").sort_values('ym')
)
print(f"五因子数据整理完成，共{len(ff5_clean)}条记录")

# 计算五因子描述性统计
factor_stats = []
for factor in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
    series = ff5_clean[factor]
    factor_stats.append({
        '变量': factor, '均值': f"{series.mean():.3f}", '标准差': f"{series.std():.3f}",
        '最小值': f"{series.min():.3f}", '中位数': f"{series.median():.3f}",
        '最大值': f"{series.max():.3f}", '观测数': len(series)
    })

factor_stats_df = pd.DataFrame(factor_stats)
factor_stats_df.to_excel(OUTPUT_DIR / "论文表2_五因子描述性统计.xlsx", index=False)
print("表2已保存")

#%% 5. 表3：因子相关性分析
def calculate_correlation_matrix_with_significance(data, factors):
    """计算带显著性的相关系数矩阵"""
    n = len(data)
    corr_matrix = pd.DataFrame(index=factors, columns=factors, dtype=object)
    p_matrix = pd.DataFrame(index=factors, columns=factors, dtype=float)
    
    for i, factor1 in enumerate(factors):
        for j, factor2 in enumerate(factors):
            if i == j:
                corr_matrix.loc[factor1, factor2] = "1.000"
                p_matrix.loc[factor1, factor2] = 0
            else:
                valid_data = data[[factor1, factor2]].dropna()
                if len(valid_data) > 2:
                    corr_coef, p_value = pearsonr(valid_data[factor1], valid_data[factor2])
                    stars = ""
                    if p_value <= 0.01: stars = "***"
                    elif p_value <= 0.05: stars = "**"
                    elif p_value <= 0.1: stars = "*"
                    if i > j:
                        corr_matrix.loc[factor1, factor2] = f"{corr_coef:.3f}{stars}"
                    else:
                        corr_matrix.loc[factor1, factor2] = ""
                    p_matrix.loc[factor1, factor2] = p_value
                else:
                    corr_matrix.loc[factor1, factor2] = ""
                    p_matrix.loc[factor1, factor2] = 1.0
    return corr_matrix, p_matrix

factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
corr_matrix_sig, p_matrix = calculate_correlation_matrix_with_significance(ff5_clean, factors)
corr_matrix_sig.to_excel(OUTPUT_DIR / "论文表3_因子相关系数.xlsx")
print("表3已保存")

#%% 6. 核心功能完成提示
print("=" * 60)
print("核心数据分析已完成！")
print("已生成以下结果文件：")
print("1. 论文表1_基于市值与账面市值比的分组收益率.xlsx")
print("2. 论文表2_五因子描述性统计.xlsx") 
print("3. 论文表3_因子相关系数.xlsx")
print("=" * 60)