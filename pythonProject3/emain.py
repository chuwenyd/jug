# 电力需求预测回归模型
# 实现多种回归算法并进行性能对比

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 将timestamp转换为datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 提取时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 引入滞后特征 (Lag Features)
    # Lag 1: 上一个半小时的负荷
    df['lag_1'] = df['electricity_demand'].shift(1)
    # Lag 48: 昨天的同一时刻 (假设数据是30分钟间隔，一天48个点)
    df['lag_48'] = df['electricity_demand'].shift(48)
    
    # 去除因 shift 产生的空值
    df.dropna(inplace=True)
    
    # 设置timestamp为索引
    df.set_index('timestamp', inplace=True)
    
    return df

# 2. 特征选择与数据集划分
def prepare_data(df):
    """准备训练和测试数据"""
    # 定义特征和目标变量
    features = ['hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend', 'lag_1', 'lag_48']
    target = 'electricity_demand'
    
    X = df[features]
    y = df[target]
    
    # 划分训练集和测试集（80%训练，20%测试）
    # 修正：使用shuffle=False保持时间序列顺序，防止数据泄露
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler

# 3. 模型训练与评估
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """训练多种回归模型并进行评估"""
    # 定义模型
    models = {
        '线性回归': LinearRegression(),
        '随机森林回归': RandomForestRegressor(n_estimators=100, random_state=42),
        '梯度提升回归': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # 训练和评估模型
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 保存结果
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        predictions[name] = y_pred
    
    return results, predictions, models

# 4. 结果可视化
def visualize_results(results, predictions, y_test, features, models, scaler, df):
    """可视化模型性能和预测结果"""
    # 1. 性能指标对比
    metrics = list(results[list(results.keys())[0]].keys())
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [results[model][metric] for model in model_names]
        bars = plt.bar(model_names, values)
        
        # 将R2转换为LaTeX格式
        if metric == 'R2':
            metric_title = r'$R^2$'
        else:
            metric_title = metric
            
        # plt.title(metric_title)  # 删除标题
        plt.ylabel(metric_title)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()
    
    # 2. 预测值与真实值对比（前100个样本）
    plt.figure(figsize=(15, 6))
    plt.plot(range(100), y_test.iloc[:100].values, label='真实值', marker='o')
    
    for name, y_pred in predictions.items():
        plt.plot(range(100), y_pred[:100], label=name, marker='x')
    
    # plt.title('模型预测值与真实值对比（前100个样本）')  # 删除标题
    plt.xlabel('样本索引')
    plt.ylabel('电力需求')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison.png')
    plt.show()
    
    # 3. 特征重要性
    for name, model in models.items():
        plt.figure(figsize=(10, 6))
        
        if hasattr(model, 'feature_importances_'):
            # 树模型的特征重要性
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            # plt.title(f'{name} - 特征重要性')  # 删除标题
            bars = plt.bar(range(len(features)), importances[indices])
            plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
        elif hasattr(model, 'coef_'):
            # 线性模型的系数重要性（绝对值）
            importances = np.abs(model.coef_)
            indices = np.argsort(importances)[::-1]
            # plt.title(f'{name} - 特征系数绝对值（重要性）')  # 删除标题
            bars = plt.bar(range(len(features)), importances[indices])
            plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{name}_feature_importance.png')
        plt.show()
    
    # 4. GBR模型预测值与真实值对比图（选取测试集中具有代表性的一周）
    if '梯度提升回归' in models and '梯度提升回归' in predictions:
        # 获取测试集对应的时间索引
        test_indices = y_test.index
        # 选取连续的7天（168小时，336个样本）
        sample_period = test_indices[:336]
        sample_y_test = y_test.loc[sample_period]
        sample_y_pred = predictions['梯度提升回归'][:336]
        
        plt.figure(figsize=(15, 8))
        plt.plot(sample_period, sample_y_test.values, label='真实值 (Ground Truth)', linewidth=2)
        plt.plot(sample_period, sample_y_pred, label='预测值 (Prediction)', linewidth=2, alpha=0.8)
        plt.xlabel('时间')
        plt.ylabel('电力负荷值')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('gbr_predicted_vs_actual.png')
        plt.show()
    
    # 5. 残差分析图
    if '梯度提升回归' in predictions:
        y_pred = predictions['梯度提升回归']
        residuals = y_test.values - y_pred
        
        # 残差随时间变化图
        plt.figure(figsize=(15, 6))
        plt.plot(y_test.index, residuals, label='残差', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('时间')
        plt.ylabel('残差值')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('residuals_over_time.png')
        plt.show()
        
        # 残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('残差值')
        plt.ylabel('频率')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('residuals_histogram.png')
        plt.show()
        
        # 残差Q-Q图
        plt.figure(figsize=(10, 6))
        sm.qqplot(residuals, line='s')
        plt.tight_layout()
        plt.savefig('residuals_qqplot.png')
        plt.show()
    
    # 6. 滑动窗口与滞后特征构造示意图
    plt.figure(figsize=(12, 6))
    
    # 绘制原始时间序列
    t = np.arange(10)
    y = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    
    plt.plot(t, y, 'o-', label='原始时间序列')
    
    # 标记滞后特征
    plt.axvline(x=5, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=4, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=5-48, color='b', linestyle='--', alpha=0.7)
    
    # 添加文本说明
    plt.text(5.1, 13, '$y_t$ (标签)', fontsize=12, color='r')
    plt.text(4.1, 11, '$y_{t-1}$ (Lag-1)', fontsize=12, color='g')
    plt.text(5-48+0.1, 2, '$y_{t-48}$ (Lag-48)', fontsize=12, color='b')
    
    # 绘制特征向量构造
    plt.arrow(4, 11, 1, 2, width=0.1, head_width=0.3, head_length=0.5, color='g')
    plt.arrow(5-48, 2, 48, 11, width=0.1, head_width=0.3, head_length=0.5, color='b')
    
    plt.xlabel('时间步')
    plt.ylabel('电力负荷值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lag_features_construction.png')
    plt.show()
    
    # 7. 自相关系数图 (ACF/PACF)
    # 绘制ACF图
    plt.figure(figsize=(15, 6))
    plot_acf(df['electricity_demand'], lags=100, ax=plt.gca())
    plt.axvline(x=48, color='r', linestyle='--', alpha=0.7)
    plt.text(48.5, 1, 'Lag-48', fontsize=12, color='r')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('acf_plot.png')
    plt.show()
    
    # 绘制PACF图
    plt.figure(figsize=(15, 6))
    plot_pacf(df['electricity_demand'], lags=100, ax=plt.gca())
    plt.axvline(x=48, color='r', linestyle='--', alpha=0.7)
    plt.text(48.5, 1, 'Lag-48', fontsize=12, color='r')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pacf_plot.png')
    plt.show()

# 5. 主函数
def main():
    # 加载和预处理数据
    df = load_and_preprocess_data('electricity_demand.csv')
    print(f'数据加载完成，共 {len(df)} 条记录')
    
    # 准备数据
    X_train, X_test, y_train, y_test, features, scaler = prepare_data(df)
    print(f'数据集划分完成，训练集: {X_train.shape[0]} 条，测试集: {X_test.shape[0]} 条')
    
    # 训练和评估模型
    results, predictions, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 打印结果
    print('\n模型性能对比：')
    print('-' * 80)
    for name, metrics in results.items():
        print(f'\n{name}:')
        for metric, value in metrics.items():
            print(f'  {metric}: {value:.4f}')
    
    # 可视化结果
    visualize_results(results, predictions, y_test, features, models, scaler, df)
    
    print('\n回归分析完成！所有结果已保存为图片文件。')

if __name__ == '__main__':
    main()
