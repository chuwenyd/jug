import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, 
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与探索
def load_data():
    df = pd.read_csv('Placement_Data_Full_Class.csv')
    print("数据集基本信息：")
    print(df.info())
    print("\n数据集前5行：")
    print(df.head())
    print("\n数据集描述性统计：")
    print(df.describe())
    print("\n缺失值情况：")
    print(df.isnull().sum())
    return df

# 2. 数据清洗
def data_cleaning(df):
    # 检查异常值
    print("\n已就业学生薪资分布：")
    print(df[df['salary'].notnull()]['salary'].describe())
    
    # 移除sl_no字段（序号，对预测无帮助）
    df = df.drop('sl_no', axis=1)
    
    return df

# 3. 特征工程
def feature_engineering(df):
    # 创建薪资区间特征
    # 根据薪资分布划分为三个区间：低薪(200000-240000)、中薪(240001-300000)、高薪(300001+)
    def salary_to_range(salary):
        if pd.isnull(salary):
            return -1  # 未就业
        elif salary <= 240000:
            return 0   # 低薪
        elif salary <= 300000:
            return 1   # 中薪
        else:
            return 2   # 高薪
    
    df['salary_range'] = df['salary'].apply(salary_to_range)
    
    # 分类变量编码
    categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # 特征相关性分析
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性热力图')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    print("\n与status的相关性：")
    print(correlation['status'].sort_values(ascending=False))
    
    print("\n与salary_range的相关性：")
    print(correlation['salary_range'].sort_values(ascending=False))
    
    # 薪资区间分布
    print("\n薪资区间分布：")
    print(df['salary_range'].value_counts())
    
    return df, le_dict

# 4. 数据准备
def prepare_data(df):
    # 划分就业预测和薪资区间预测的数据集
    # 就业预测：所有学生
    X_placement = df.drop(['status', 'salary', 'salary_range'], axis=1)
    y_placement = df['status']
    
    # 薪资区间预测：仅已就业的学生
    df_placed = df[df['status'] == 1]
    X_salary = df_placed.drop(['status', 'salary', 'salary_range'], axis=1)
    y_salary = df_placed['salary_range']
    
    # 薪资二分类预测：仅已就业的学生
    # 使用原始薪资值进行二分类
    y_salary_binary = df_placed['salary']
    # 计算薪资中位数
    median_val = y_salary_binary.median()
    print(f"\n薪资中位数：{median_val:.2f}")
    
    # 转换为二分类：1表示高于中位数，0表示低于等于中位数
    y_salary_binary = (y_salary_binary > median_val).astype(int)
    print(f"\n薪资二分类分布：")
    print(y_salary_binary.value_counts())
    
    # 划分训练集和测试集
    X_train_placement, X_test_placement, y_train_placement, y_test_placement = train_test_split(
        X_placement, y_placement, test_size=0.2, random_state=42
    )
    
    X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
        X_salary, y_salary, test_size=0.2, random_state=42
    )
    
    # 薪资二分类训练集和测试集划分
    X_train_salary_binary, X_test_salary_binary, y_train_salary_binary, y_test_salary_binary = train_test_split(
        X_salary, y_salary_binary, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler_placement = StandardScaler()
    X_train_placement_scaled = scaler_placement.fit_transform(X_train_placement)
    X_test_placement_scaled = scaler_placement.transform(X_test_placement)
    
    scaler_salary = StandardScaler()
    X_train_salary_scaled = scaler_salary.fit_transform(X_train_salary)
    X_test_salary_scaled = scaler_salary.transform(X_test_salary)
    
    # 薪资二分类使用相同的标准化器
    X_train_salary_binary_scaled = scaler_salary.transform(X_train_salary_binary)
    X_test_salary_binary_scaled = scaler_salary.transform(X_test_salary_binary)
    
    return {
        'placement': {
            'X_train': X_train_placement_scaled,
            'X_test': X_test_placement_scaled,
            'y_train': y_train_placement,
            'y_test': y_test_placement
        },
        'salary': {
            'X_train': X_train_salary_scaled,
            'X_test': X_test_salary_scaled,
            'y_train': y_train_salary,
            'y_test': y_test_salary
        },
        'salary_binary': {
            'X_train': X_train_salary_binary_scaled,
            'X_test': X_test_salary_binary_scaled,
            'y_train': y_train_salary_binary,
            'y_test': y_test_salary_binary,
            'median': median_val
        }
    }

# 5. 模型训练与评估 - 就业预测
def train_placement_models(data):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # 训练模型
        model.fit(data['X_train'], data['y_train'])
        
        # 预测
        y_pred = model.predict(data['X_test'])
        
        # 评估指标
        accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred)
        recall = recall_score(data['y_test'], y_pred)
        f1 = f1_score(data['y_test'], y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
        
        # 打印混淆矩阵
        cm = confusion_matrix(data['y_test'], y_pred)
        print(f"\n{name} 混淆矩阵：")
        print(cm)
    
    return results

# 6. 模型训练与评估 - 薪资区间预测
def train_salary_models(data):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, multi_class='ovr'),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, decision_function_shape='ovr'),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # 训练模型
        model.fit(data['X_train'], data['y_train'])
        
        # 预测
        y_pred = model.predict(data['X_test'])
        
        # 评估指标
        accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred, average='weighted')
        recall = recall_score(data['y_test'], y_pred, average='weighted')
        f1 = f1_score(data['y_test'], y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
        
        # 打印混淆矩阵
        cm = confusion_matrix(data['y_test'], y_pred)
        print(f"\n{name} 混淆矩阵：")
        print(cm)
    
    return results

# 7. 模型训练与评估 - 薪资二分类预测
def train_salary_binary_models(data):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # 训练模型
        model.fit(data['X_train'], data['y_train'])
        
        # 预测
        y_pred = model.predict(data['X_test'])
        
        # 评估指标
        accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred)
        recall = recall_score(data['y_test'], y_pred)
        f1 = f1_score(data['y_test'], y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
        
        # 打印混淆矩阵
        cm = confusion_matrix(data['y_test'], y_pred)
        print(f"\n{name} 混淆矩阵：")
        print(cm)
    
    return results

# 8. 结果展示与比较
def display_results(placement_results, salary_results, salary_binary_results, data):
    # 就业预测模型比较
    print("\n" + "="*50)
    print("就业预测模型性能比较")
    print("="*50)
    
    placement_df = pd.DataFrame.from_dict(
        {name: [result['accuracy'], result['precision'], result['recall'], result['f1']]
         for name, result in placement_results.items()},
        orient='index',
        columns=['Accuracy', 'Precision', 'Recall', 'F1 Score']
    )
    
    print(placement_df.sort_values('F1 Score', ascending=False))
    
    # 可视化就业预测模型性能
    plt.figure(figsize=(12, 8))
    placement_df.plot(kind='bar', figsize=(12, 8))
    plt.title('就业预测模型性能比较')
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('placement_models_comparison.png')
    plt.close()
    
    # 薪资区间预测模型比较
    print("\n" + "="*50)
    print("薪资区间预测模型性能比较")
    print("="*50)
    
    salary_df = pd.DataFrame.from_dict(
        {name: [result['accuracy'], result['precision'], result['recall'], result['f1']]
         for name, result in salary_results.items()},
        orient='index',
        columns=['Accuracy', 'Precision', 'Recall', 'F1 Score']
    )
    
    print(salary_df.sort_values('F1 Score', ascending=False))
    
    # 可视化薪资区间预测模型性能
    plt.figure(figsize=(12, 8))
    salary_df.plot(kind='bar', figsize=(12, 8))
    plt.title('薪资区间预测模型性能比较')
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('salary_models_comparison.png')
    plt.close()
    
    # 薪资二分类预测模型比较
    print("\n" + "="*50)
    print("薪资二分类预测模型性能比较")
    print("="*50)
    
    salary_binary_df = pd.DataFrame.from_dict(
        {name: [result['accuracy'], result['precision'], result['recall'], result['f1']]
         for name, result in salary_binary_results.items()},
        orient='index',
        columns=['Accuracy', 'Precision', 'Recall', 'F1 Score']
    )
    
    print(salary_binary_df.sort_values('F1 Score', ascending=False))
    
    # 可视化薪资二分类预测模型性能
    plt.figure(figsize=(12, 8))
    salary_binary_df.plot(kind='bar', figsize=(12, 8))
    plt.title('薪资二分类预测模型性能比较')
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('salary_binary_models_comparison.png')
    plt.close()
    
    # 最佳模型详细报告
    best_placement_model = max(placement_results.items(), key=lambda x: x[1]['f1'])
    print("\n" + "="*50)
    print(f"最佳就业预测模型: {best_placement_model[0]}")
    print("="*50)
    print(f"Accuracy: {best_placement_model[1]['accuracy']:.4f}")
    print(f"Precision: {best_placement_model[1]['precision']:.4f}")
    print(f"Recall: {best_placement_model[1]['recall']:.4f}")
    print(f"F1 Score: {best_placement_model[1]['f1']:.4f}")
    print("\n分类报告:")
    print(classification_report(data['placement']['y_test'], best_placement_model[1]['y_pred']))
    
    best_salary_model = max(salary_results.items(), key=lambda x: x[1]['f1'])
    print("\n" + "="*50)
    print(f"最佳薪资区间预测模型: {best_salary_model[0]}")
    print("="*50)
    print(f"Accuracy: {best_salary_model[1]['accuracy']:.4f}")
    print(f"Precision: {best_salary_model[1]['precision']:.4f}")
    print(f"Recall: {best_salary_model[1]['recall']:.4f}")
    print(f"F1 Score: {best_salary_model[1]['f1']:.4f}")
    print("\n分类报告:")
    print(classification_report(data['salary']['y_test'], best_salary_model[1]['y_pred']))
    
    best_salary_binary_model = max(salary_binary_results.items(), key=lambda x: x[1]['f1'])
    print("\n" + "="*50)
    print(f"最佳薪资二分类预测模型: {best_salary_binary_model[0]}")
    print("="*50)
    print(f"Accuracy: {best_salary_binary_model[1]['accuracy']:.4f}")
    print(f"Precision: {best_salary_binary_model[1]['precision']:.4f}")
    print(f"Recall: {best_salary_binary_model[1]['recall']:.4f}")
    print(f"F1 Score: {best_salary_binary_model[1]['f1']:.4f}")
    print("\n分类报告:")
    print(classification_report(data['salary_binary']['y_test'], best_salary_binary_model[1]['y_pred']))
    
    # 薪资区间分布可视化
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data['salary']['y_test'], palette='viridis')
    plt.title('测试集真实薪资区间分布')
    plt.xlabel('薪资区间')
    plt.ylabel('学生人数')
    plt.xticks([0, 1, 2], ['低薪(≤240000)', '中薪(240001-300000)', '高薪(>300000)'])
    plt.tight_layout()
    plt.savefig('salary_range_distribution.png')
    plt.close()
    
    # 薪资二分类分布可视化
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data['salary_binary']['y_test'], palette='viridis')
    plt.title('测试集真实薪资二分类分布')
    plt.xlabel('薪资类别')
    plt.ylabel('学生人数')
    plt.xticks([0, 1], ['低于等于中位数', '高于中位数'])
    plt.tight_layout()
    plt.savefig('salary_binary_distribution.png')
    plt.close()

# 主函数
def main():
    print("="*50)
    print("学生就业与薪资预测分析")
    print("="*50)
    
    # 加载数据
    df = load_data()
    
    # 数据清洗
    df = data_cleaning(df)
    
    # 特征工程
    df, le_dict = feature_engineering(df)
    
    # 数据准备
    data = prepare_data(df)
    
    # 模型训练与评估
    print("\n" + "="*50)
    print("训练就业预测模型...")
    print("="*50)
    placement_results = train_placement_models(data['placement'])
    
    print("\n" + "="*50)
    print("训练薪资区间预测模型...")
    print("="*50)
    salary_results = train_salary_models(data['salary'])
    
    print("\n" + "="*50)
    print("训练薪资二分类预测模型...")
    print("="*50)
    salary_binary_results = train_salary_binary_models(data['salary_binary'])
    
    # 结果展示
    display_results(placement_results, salary_results, salary_binary_results, data)
    
    print("\n" + "="*50)
    print("分析完成！")
    print("="*50)
    print("生成的文件：")
    print("- correlation_heatmap.png: 特征相关性热力图")
    print("- placement_models_comparison.png: 就业预测模型比较图")
    print("- salary_models_comparison.png: 薪资区间预测模型比较图")
    print("- salary_binary_models_comparison.png: 薪资二分类预测模型比较图")
    print("- salary_range_distribution.png: 薪资区间分布可视化图")
    print("- salary_binary_distribution.png: 薪资二分类分布可视化图")

if __name__ == "__main__":
    main()