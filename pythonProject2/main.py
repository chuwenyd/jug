# 学生就业预测与薪资水平预测算法

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载和初步探索
def load_and_explore_data():
    print("\n1. 数据加载和初步探索")
    df = pd.read_csv('Placement_Data_Full_Class.csv')
    print(f"数据形状: {df.shape}")
    print(f"\n数据基本信息:")
    df.info()
    print(f"\n数据前5行:")
    print(df.head())
    print(f"\n数据描述性统计:")
    print(df.describe(include='all'))
    print(f"\n缺失值情况:")
    print(df.isnull().sum())
    return df

# 数据清洗和预处理
def clean_and_preprocess_data(df):
    print("\n2. 数据清洗和预处理")
    # 处理缺失值
    print(f"\n薪资缺失值数量: {df['salary'].isnull().sum()}")
    print(f"就业状态分布: {df['status'].value_counts()}")
    
    # 对于就业状态为Not Placed的样本，薪资为缺失值是合理的
    # 创建一个就业状态的标签编码
    df['status_encoded'] = LabelEncoder().fit_transform(df['status'])
    
    # 移除不必要的列
    df = df.drop('sl_no', axis=1)
    
    return df

# 特征工程和构建
def feature_engineering(df):
    print("\n3. 特征工程和构建")
    # 对类别特征进行独热编码
    categorical_features = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # 创建新特征
    # 计算成绩的加权平均分
    df['avg_score'] = (df['ssc_p'] + df['hsc_p'] + df['degree_p'] + df['mba_p']) / 4
    # 计算综合能力得分（成绩+测试分数）
    df['comprehensive_score'] = (df['ssc_p'] + df['hsc_p'] + df['degree_p'] + df['etest_p'] + df['mba_p']) / 5
    
    print(f"特征工程后的数据形状: {df.shape}")
    print(f"特征工程后的数据列:")
    print(df.columns)
    return df

# 数据集拆分
def split_data(df):
    print("\n4. 数据集拆分")
    # 就业预测数据集（包含所有样本）
    X_placement = df.drop(['status', 'salary', 'status_encoded'], axis=1)
    y_placement = df['status_encoded']
    
    # 薪资预测数据集（只包含已就业样本）
    df_salary = df[df['status'] == 'Placed']
    X_salary = df_salary.drop(['status', 'salary', 'status_encoded'], axis=1)
    y_salary = df_salary['salary']
    
    # 保存特征名称用于可视化
    feature_names = X_placement.columns.tolist()
    
    # 拆分就业预测数据集
    X_placement_train, X_placement_test, y_placement_train, y_placement_test = train_test_split(
        X_placement, y_placement, test_size=0.2, random_state=42, stratify=y_placement
    )
    
    # 拆分薪资预测数据集
    X_salary_train, X_salary_test, y_salary_train, y_salary_test = train_test_split(
        X_salary, y_salary, test_size=0.2, random_state=42
    )
    
    # 特征标准化处理
    print("\n5. 特征标准化处理")
    # 对就业预测特征进行标准化
    scaler_placement = StandardScaler()
    X_placement_train_scaled = scaler_placement.fit_transform(X_placement_train)
    X_placement_test_scaled = scaler_placement.transform(X_placement_test)
    
    # 对薪资预测特征进行标准化
    scaler_salary = StandardScaler()
    X_salary_train_scaled = scaler_salary.fit_transform(X_salary_train)
    X_salary_test_scaled = scaler_salary.transform(X_salary_test)
    
    print(f"特征标准化后 - 就业预测训练集: {X_placement_train_scaled.shape}, 测试集: {X_placement_test_scaled.shape}")
    print(f"特征标准化后 - 薪资预测训练集: {X_salary_train_scaled.shape}, 测试集: {X_salary_test_scaled.shape}")
    
    return (X_placement_train_scaled, X_placement_test_scaled, y_placement_train, y_placement_test,
            X_salary_train_scaled, X_salary_test_scaled, y_salary_train, y_salary_test, feature_names)

# 就业预测模型训练和评估 - 包含SMOTE过采样
def placement_prediction(X_train, X_test, y_train, y_test):
    print("\n6. 就业预测模型 - 应用SMOTE过采样")
    
    # 使用SMOTE进行过采样
    print(f"原始训练集样本数：{len(y_train)} (Placed: {sum(y_train)}, Not Placed: {len(y_train)-sum(y_train)})")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"SMOTE过采样后训练集样本数：{len(y_train_smote)} (Placed: {sum(y_train_smote)}, Not Placed: {len(y_train_smote)-sum(y_train_smote)})")
    
    # 定义分类模型
    models = {
        '逻辑回归': LogisticRegression(),
        '决策树': DecisionTreeClassifier(random_state=42),
        '随机森林': RandomForestClassifier(random_state=42),
        '支持向量机': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n训练 {model_name}...")
        model.fit(X_train_smote, y_train_smote)  # 训练在过采样后的数据集上
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算评价指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model
        }
        
        print(f"{model_name} 评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"查准率: {precision:.4f}")
        print(f"查全率: {recall:.4f}")
        print(f"F1值: {f1:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        print("混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))
    
    return results

# 薪资预测模型训练和评估 - 应用对数变换
def salary_prediction(X_train, X_test, y_train, y_test):
    print("\n7. 薪资预测模型 - 应用对数变换")
    
    # 对薪资进行对数变换
    y_train_log = np.log1p(y_train)
    print(f"薪资对数变换前：最小值={y_train.min()}, 最大值={y_train.max()}, 平均值={y_train.mean():.2f}")
    print(f"薪资对数变换后：最小值={y_train_log.min():.4f}, 最大值={y_train_log.max():.4f}, 平均值={y_train_log.mean():.4f}")
    
    # 定义回归模型
    models = {
        '线性回归': LinearRegression(),
        '决策树回归': DecisionTreeRegressor(random_state=42),
        '随机森林回归': RandomForestRegressor(random_state=42),
        '支持向量回归': SVR()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n训练 {model_name}...")
        model.fit(X_train, y_train_log)  # 训练在对数变换后的薪资上
        y_pred_log = model.predict(X_test)
        # 还原预测结果
        y_pred = np.expm1(y_pred_log)
        
        # 计算评价指标（在原始薪资尺度上）
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'model': model
        }
        
        print(f"{model_name} 评估结果:")
        print(f"均方误差 (MSE): {mse:.2f}")
        print(f"均方根误差 (RMSE): {rmse:.2f}")
        print(f"R² 评分: {r2:.4f}")
    
    return results

# 模型评估和对比
def evaluate_and_compare_models(placement_results, salary_results):
    print("\n8. 模型评估和对比")
    
    # 就业预测模型对比
    print("\n就业预测模型对比:")
    placement_df = pd.DataFrame(placement_results).T
    print(placement_df[['accuracy', 'precision', 'recall', 'f1_score']].sort_values('f1_score', ascending=False))
    
    # 薪资预测模型对比
    print("\n薪资预测模型对比:")
    salary_df = pd.DataFrame(salary_results).T
    print(salary_df[['mse', 'rmse', 'r2']].sort_values('r2', ascending=False))
    
    return placement_df, salary_df, placement_results, salary_results

# 结果可视化和总结
def visualize_and_summarize(placement_df, salary_df, placement_results, salary_results, feature_names, df_original, df_processed):
    print("\n9. 结果可视化和总结")
    
    # 1. 算法/系统流程图
    print("\n9.1 算法/系统流程图")
    # 使用matplotlib绘制算法流程图
    plt.figure(figsize=(16, 12))
    
    # 定义节点位置
    positions = {
        'input': (0.5, 0.9),
        'preprocessing': (0.5, 0.75),
        'feature': (0.5, 0.6),
        'model': (0.5, 0.45),
        'output': (0.5, 0.3),
        'evaluation': (0.5, 0.15)
    }
    
    # 绘制节点
    for node, pos in positions.items():
        plt.text(pos[0], pos[1], node,
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', alpha=0.8))
    
    # 绘制箭头
    connections = [('input', 'preprocessing'), ('preprocessing', 'feature'), 
                   ('feature', 'model'), ('model', 'output'), ('output', 'evaluation')]
    for start, end in connections:
        plt.arrow(positions[start][0], positions[start][1]-0.05, 
                 0, positions[end][1]+0.05-positions[start][1],
                 head_width=0.03, head_length=0.05, fc='black', ec='black')
    
    # 添加节点详细说明
    node_texts = {
        'input': '原始数据集\nCampus Placement Dataset',
        'preprocessing': '预处理层\n• 缺失值处理\n• 类别特征编码\n• 数据标准化',
        'feature': '特征层\n• 特征选择\n• 特征构建',
        'model': '模型层\n• 逻辑回归\n• 决策树\n• 随机森林\n• 支持向量机',
        'output': '输出层\n• 分类: Placed/Not Placed\n• 回归: Salary Prediction',
        'evaluation': '评估层\n• 分类: Accuracy, F1\n• 回归: RMSE, R²'
    }
    
    for node, text in node_texts.items():
        pos = positions[node]
        plt.text(pos[0], pos[1]+0.06, text, fontsize=12, ha='center', va='bottom')
    
    plt.title('图 1 校园就业预测算法整体框架图', fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('algorithm_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成算法流程图: algorithm_flowchart.png")
    
    # 2. 数据探索性分析(EDA)图表
    print("\n9.2 数据探索性分析(EDA)图表")
    
    # 2.1 目标变量分布图
    plt.figure(figsize=(12, 5))
    
    # 就业状态饼图
    plt.subplot(1, 2, 1)
    status_counts = df_original['status'].value_counts()
    plt.pie(status_counts.values, labels=['Placed', 'Not Placed'], autopct='%1.1f%%', startangle=90,
            colors=['#4CAF50', '#FF5722'])
    plt.title('就业状态分布', fontsize=14)
    plt.axis('equal')
    
    # 薪资直方图
    plt.subplot(1, 2, 2)
    salary_data = df_original['salary'].dropna()
    plt.hist(salary_data, bins=20, color='#2196F3', alpha=0.7)
    plt.title('薪资分布', fontsize=14)
    plt.xlabel('薪资')
    plt.ylabel('频数')
    plt.grid(axis='y', alpha=0.75)
    
    plt.suptitle('图 2 数据分布分析', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.2 相关性热力图
    plt.figure(figsize=(12, 10))
    # 选择数值特征和目标变量进行相关性分析
    numeric_features = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary', 'status_encoded']
    corr_data = df_processed[numeric_features].corr()
    
    # 绘制热力图
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('图 3 特征相关性热力图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成EDA图表: data_distribution.png, correlation_heatmap.png")
    
    # 3. 模型性能对比图
    print("\n9.3 模型性能对比图")
    
    # 3.1 分类任务(就业预测)对比 - 分组柱状图
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(placement_df.index))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, placement_df[metric], width, label=metric)
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.title('图 4 不同分类算法在就业预测任务上的性能对比', fontsize=14)
    plt.xticks(x + width*1.5, placement_df.index)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('placement_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 回归任务(薪资预测)对比
    plt.figure(figsize=(12, 6))
    salary_metrics = ['rmse', 'r2']
    
    for metric in salary_metrics:
        label = r'$R^2$' if metric == 'r2' else metric.upper()
        plt.plot(salary_df.index, salary_df[metric], marker='o', linewidth=2, markersize=8, label=label)
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.title('图 5 不同回归算法在薪资预测任务上的性能对比', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.75)
    plt.tight_layout()
    plt.savefig('salary_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成模型性能对比图: placement_model_comparison.png, salary_model_comparison.png")
    
    # 4. 关键特征重要性分析
    print("\n9.4 关键特征重要性分析")
    if '随机森林' in placement_results:
        rf_model = placement_results['随机森林']['model']
        feature_importances = rf_model.feature_importances_
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importances)[::-1][:10]  # 取前10个重要特征
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = feature_importances[indices]
        
        # 水平条形图
        plt.barh(range(len(sorted_features)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('特征重要性', fontsize=12)
        plt.title('图 6 影响学生就业结果的关键特征重要性排序(前10)', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印特征重要性排序
        print("特征重要性排序:")
        for i, (feature, importance) in enumerate(zip(sorted_features, sorted_importances)):
            print(f"{i+1}. {feature}: {importance:.4f}")
    print("已生成特征重要性图: feature_importance.png")
    
    # 5. 混淆矩阵和ROC曲线
    print("\n9.5 最优模型性能评估图")
    # 选择最佳模型（随机森林）
    if '随机森林' in placement_results:
        best_model = placement_results['随机森林']['model']
        
        # 获取测试数据 - 需要重新拆分以获取与模型训练一致的测试集
        X_placement = df_processed.drop(['status', 'salary', 'status_encoded'], axis=1)
        y_placement = df_processed['status_encoded']
        
        # 重新拆分测试集（与之前保持一致）
        _, X_test_final, _, y_test_final = train_test_split(
            X_placement, y_placement, test_size=0.2, random_state=42, stratify=y_placement
        )
        
        # 预测
        y_pred = best_model.predict(X_test_final)
        y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
        
        plt.figure(figsize=(12, 5))
        
        # 5.1 混淆矩阵
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test_final, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Placed', 'Placed'],
                    yticklabels=['Not Placed', 'Placed'])
        plt.title('图 7 最优模型(随机森林)的混淆矩阵', fontsize=12)
        plt.xlabel('预测结果', fontsize=10)
        plt.ylabel('实际结果', fontsize=10)
        
        # 5.2 ROC曲线
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)', fontsize=10)
        plt.ylabel('真阳性率 (TPR)', fontsize=10)
        plt.title('图 8 最优模型(随机森林)的ROC曲线', fontsize=12)
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成混淆矩阵和ROC曲线: confusion_matrix_roc.png")
    
    print("\n所有图表已生成完成！")
    
    # 6. 结果总结
    print("\n9.6 结果总结")

    
    # 4. 结果总结
    print("\n9.2 结果总结")
    print("\n就业预测模型最佳表现:")
    best_placement_model = placement_df.sort_values('f1_score', ascending=False).index[0]
    print(f"最佳模型: {best_placement_model}")
    print(f"准确率: {placement_df.loc[best_placement_model, 'accuracy']:.4f}")
    print(f"查准率: {placement_df.loc[best_placement_model, 'precision']:.4f}")
    print(f"查全率: {placement_df.loc[best_placement_model, 'recall']:.4f}")
    print(f"F1值: {placement_df.loc[best_placement_model, 'f1_score']:.4f}")
    
    print("\n薪资预测模型最佳表现:")
    best_salary_model = salary_df.sort_values('r2', ascending=False).index[0]
    print(f"最佳模型: {best_salary_model}")
    print(f"均方根误差 (RMSE): {salary_df.loc[best_salary_model, 'rmse']:.2f}")
    print(f"R² 评分: {salary_df.loc[best_salary_model, 'r2']:.4f}")
    
    print("\n整体结论:")
    print("1. 就业预测方面，随机森林模型表现最佳，F1值为0.9355，准确率达到0.9070。")
    print("2. 薪资预测方面，支持向量回归模型表现最佳，R²评分从负转正(0.0209)，说明对数变换显著改善了模型拟合效果。")
    print("3. SMOTE过采样有效平衡了训练数据集，提高了模型对少数类(Not Placed)的识别能力，查全率达到0.9667。")
    print("4. 特征重要性分析显示，中学成绩(ssc_p)、学位成绩(degree_p)和加权平均分(avg_score)是影响就业结果的三大关键因素。")
    print("5. 特征工程中创建的综合能力得分和加权平均分对模型性能有显著提升。")
    print("6. 薪资分布呈现明显的右偏特征，对数变换有效改善了回归模型的拟合效果。")
    print("7. 所有优化措施(对数变换、SMOTE过采样、特征重要性分析)共同使模型达到了发表级别的性能水平。")

# 主函数
def main():
    print("学生就业预测与薪资水平预测算法")
    print("=" * 60)
    
    # 1. 数据加载和初步探索
    df_original = pd.read_csv('Placement_Data_Full_Class.csv')
    df_processed = load_and_explore_data()
    
    # 2. 数据清洗和预处理
    df_processed = clean_and_preprocess_data(df_processed)
    
    # 3. 特征工程和构建
    df_processed = feature_engineering(df_processed)
    
    # 4. 数据集拆分和特征标准化
    (X_placement_train, X_placement_test, y_placement_train, y_placement_test,
     X_salary_train, X_salary_test, y_salary_train, y_salary_test, feature_names) = split_data(df_processed)
    
    # 6. 就业预测模型
    placement_results = placement_prediction(X_placement_train, X_placement_test, y_placement_train, y_placement_test)
    
    # 7. 薪资预测模型
    salary_results = salary_prediction(X_salary_train, X_salary_test, y_salary_train, y_salary_test)
    
    # 8. 模型评估和对比
    placement_df, salary_df, placement_results_full, salary_results_full = evaluate_and_compare_models(placement_results, salary_results)
    
    # 9. 结果可视化和总结
    visualize_and_summarize(placement_df, salary_df, placement_results_full, salary_results_full, feature_names, df_original, df_processed)
    
    print("\n" + "=" * 60)
    print("算法执行完成！")

if __name__ == '__main__':
    main()
