import pandas as pd
import pandas as pd    #用于数据处理和分析，可处理表格数据。
import numpy as np     #用于数值计算，提供了高效的数组操作。
import matplotlib.pyplot as plt    #用于绘制各种类型的图表
import seaborn as sns   #基于matplotlib的高级绘图库，能绘制更美观的统计图形。
import warnings
warnings.filterwarnings("ignore")
 
 # 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
data = pd.read_csv('python60-days-challenge\python-learning-library\heart.csv')    #读取数据
print(data.columns)    #'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
# 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
#提取连续特征值
continuous_features = ['age','trestbps', 'chol', 'thalach','oldpeak']
print(continuous_features)    #打印出连续特征值的列名
#提取离散特征值
discrete_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target']
print(discrete_features)    #打印出离散特征值的列名

#使用映射字典进行转换
mapping_dict = {
    'cp':{0:0,1:1,2:2,3:3},
    'restecg':{0:0,1:1,2:2},
    'slope':{0:0,1:1,2:2},
    'thal':{0:0,1:1,2:2,3:3},
    'ca':{0:0,1:1,2:2,3:3,4:4}
}
for feature, mapping_dict in mapping_dict.items():    #遍历映射字典
    data[feature] = data[feature].map(mapping_dict)    #将映射字典中的值替换为原数据中的值

#对离散特征值进行独热编码
data = pd.get_dummies(data, columns=['sex', 'fbs','exang'])    #对离散特征值进行独热编码
print(data.columns)    #打印出数据的列名
data2 = pd.read_csv("python60-days-challenge\python-learning-library\heart.csv")    #读取数据
list_final = []    #新建一个空列表，用于存放独热编码后新增的特征名
for i in data.columns:    #遍历数据的列名
    if i not in data2.columns:    #如果列名不在原数据的列名中
        list_final.append(i)    #将列名添加到列表中
for i in list_final:    #遍历列表中的列名
    data[i] = data[i].astype(int)    #将列名转换为int类型
#划分训练集和测试集
from sklearn.model_selection import train_test_split    #导入train_test_split函数
X = data.drop(['target'], axis=1)    #特征，axis=1表示按列删除
y = data['target']    #标签
# #按照8:2划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    #80%训练集，20%测试集
# print(X_train.shape)    #打印出训练集的形状
# print(X_test.shape)    #打印出测试集的形状
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# 便准化数据（聚类前通常需要标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 11)  # 尝试不同的簇数
inertia_scores = []  # 存储每个簇数对应的轮廓系数
silhouette_scores = []
ch_scores = []
db_scores = []

# 尝试不同的簇数
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # 初始化KMeans模型
    kmeans_labels = kmeans.fit_predict(X_scaled)
    inertia_scores.append(kmeans.inertia_)  # 计算轮廓系数
    silhouette = silhouette_score(X_scaled, kmeans_labels)  # 轮廓系数
    silhouette_scores.append(silhouette)
    ch = calinski_harabasz_score(X_scaled, kmeans_labels)  # CH 指数
    ch_scores.append(ch)
    db = davies_bouldin_score(X_scaled, kmeans_labels)  # DB 指数
    db_scores.append(db)
    print(f"k={k}, 惯性: {kmeans.inertia_:.2f}, 轮廓系数: {silhouette:.3f}, CH 指数: {ch:.2f}, DB 指数: {db:.3f}")

plt.figure(figsize=(15, 10))

# 肘部法则图（Inertia）
plt.subplot(2, 2, 1)
plt.plot(k_range, inertia_scores, marker='o')
plt.title('肘部法则确定最优聚类数 k（惯性，越小越好）')
plt.xlabel('聚类数 (k)')
plt.ylabel('惯性')
plt.grid(True)

# 轮廓系数图
plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.title('轮廓系数确定最优聚类数 k（越大越好）')
plt.xlabel('聚类数 (k)')
plt.ylabel('轮廓系数')
plt.grid(True)

# CH 指数图
plt.subplot(2, 2, 3)
plt.plot(k_range, ch_scores, marker='o', color='green')
plt.title('Calinski-Harabasz 指数确定最优聚类数 k（越大越好）')
plt.xlabel('聚类数 (k)')
plt.ylabel('CH 指数')
plt.grid(True)

# DB 指数图
plt.subplot(2, 2, 4)
plt.plot(k_range, db_scores, marker='o', color='red')
plt.title('Davies-Bouldin 指数确定最优聚类数 k（越小越好）')
plt.xlabel('聚类数 (k)')
plt.ylabel('DB 指数')
plt.grid(True)

plt.tight_layout()
plt.show()


# 提示用户选择 k 值
selected_k = 4

# 使用选择的 k 值进行 KMeans 聚类
kmeans = KMeans(n_clusters=selected_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
X['KMeans_Cluster'] = kmeans_labels

# 使用 PCA 降维到 2D 进行可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans 聚类结果可视化
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis')
plt.title(f'KMeans Clustering with k={selected_k} (PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 打印 KMeans 聚类标签的前几行
print(f"KMeans Cluster labels (k={selected_k}) added to X:")
print(X[['KMeans_Cluster']].value_counts())

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns



# 评估不同 eps 和 min_samples 下的指标
# eps这个参数表示邻域的半径，min_samples表示一个点被认为是核心点所需的最小样本数。
# min_samples这个参数表示一个核心点所需的最小样本数。

eps_range = np.arange(0.3, 2.0, 0.1)  # 测试 eps 从 0.3 到 0.7
min_samples_range = range(3, 10)  # 测试 min_samples 从 3 到 7
results = []

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        # 计算簇的数量（排除噪声点 -1）
        n_clusters = len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        # 计算噪声点数量
        n_noise = list(dbscan_labels).count(-1)
        # 只有当簇数量大于 1 且有有效簇时才计算评估指标
        if n_clusters > 1:
            # 排除噪声点后计算评估指标
            mask = dbscan_labels != -1
            if mask.sum() > 0:  # 确保有非噪声点
                silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
                ch = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
                db = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': silhouette,
                    'ch_score': ch,
                    'db_score': db
                })
                print(f"eps={eps:.1f}, min_samples={min_samples}, 簇数: {n_clusters}, 噪声点: {n_noise}, "
                      f"轮廓系数: {silhouette:.3f}, CH 指数: {ch:.2f}, DB 指数: {db:.3f}")
        else:
            print(f"eps={eps:.1f}, min_samples={min_samples}, 簇数: {n_clusters}, 噪声点: {n_noise}, 无法计算评估指标")

# 将结果转为 DataFrame 以便可视化和选择参数
results_df = pd.DataFrame(results)

# 绘制评估指标图，增加点论文中的工作量
plt.figure(figsize=(15, 10))
# 轮廓系数图
plt.subplot(2, 2, 1)
for min_samples in min_samples_range:
    subset = results_df[results_df['min_samples'] == min_samples] # 
    plt.plot(subset['eps'], subset['silhouette'], marker='o', label=f'min_samples={min_samples}')
plt.title('轮廓系数确定最优参数（越大越好）')
plt.xlabel('eps')
plt.ylabel('轮廓系数')
plt.legend()
plt.grid(True)

# CH 指数图
plt.subplot(2, 2, 2)
for min_samples in min_samples_range:
    subset = results_df[results_df['min_samples'] == min_samples]
    plt.plot(subset['eps'], subset['ch_score'], marker='o', label=f'min_samples={min_samples}')
plt.title('Calinski-Harabasz 指数确定最优参数（越大越好）')
plt.xlabel('eps')
plt.ylabel('CH 指数')
plt.legend()
plt.grid(True)

# DB 指数图
plt.subplot(2, 2, 3)
for min_samples in min_samples_range:
    subset = results_df[results_df['min_samples'] == min_samples]
    plt.plot(subset['eps'], subset['db_score'], marker='o', label=f'min_samples={min_samples}')
plt.title('Davies-Bouldin 指数确定最优参数（越小越好）')
plt.xlabel('eps')
plt.ylabel('DB 指数')
plt.legend()
plt.grid(True)

# 簇数量图
plt.subplot(2, 2, 4)
for min_samples in min_samples_range:
    subset = results_df[results_df['min_samples'] == min_samples]
    plt.plot(subset['eps'], subset['n_clusters'], marker='o', label=f'min_samples={min_samples}')
plt.title('簇数量变化')
plt.xlabel('eps')
plt.ylabel('簇数量')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 选择 eps 和 min_samples 值（根据图表选择最佳参数）
selected_eps = 1.1  # 根据图表调整
selected_min_samples = 3# 根据图表调整

# 使用选择的参数进行 DBSCAN 聚类
dbscan = DBSCAN(eps=selected_eps, min_samples=selected_min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)
X['DBSCAN_Cluster'] = dbscan_labels

# 使用 PCA 降维到 2D 进行可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DBSCAN 聚类结果可视化
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='viridis')
plt.title(f'DBSCAN Clustering with eps={selected_eps}, min_samples={selected_min_samples} (PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 打印 DBSCAN 聚类标签的分布
print(f"DBSCAN Cluster labels (eps={selected_eps}, min_samples={selected_min_samples}) added to X:")
print(X[['DBSCAN_Cluster']].value_counts())



import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 评估不同 n_clusters 下的指标
n_clusters_range = range(2, 11)  # 测试簇数量从 2 到 10
silhouette_scores = []
ch_scores = []
db_scores = []

for n_clusters in n_clusters_range:
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')  # 使用 Ward 准则合并簇
    agglo_labels = agglo.fit_predict(X_scaled)
    
    # 计算评估指标
    silhouette = silhouette_score(X_scaled, agglo_labels)
    ch = calinski_harabasz_score(X_scaled, agglo_labels)
    db = davies_bouldin_score(X_scaled, agglo_labels)
    
    silhouette_scores.append(silhouette)
    ch_scores.append(ch)
    db_scores.append(db)
    
    print(f"n_clusters={n_clusters}, 轮廓系数: {silhouette:.3f}, CH 指数: {ch:.2f}, DB 指数: {db:.3f}")

# 绘制评估指标图
plt.figure(figsize=(15, 5))

# 轮廓系数图
plt.subplot(1, 3, 1)
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('轮廓系数确定最优簇数（越大越好）')
plt.xlabel('簇数量 (n_clusters)')
plt.ylabel('轮廓系数')
plt.grid(True)

# CH 指数图
plt.subplot(1, 3, 2)
plt.plot(n_clusters_range, ch_scores, marker='o')
plt.title('Calinski-Harabasz 指数确定最优簇数（越大越好）')
plt.xlabel('簇数量 (n_clusters)')
plt.ylabel('CH 指数')
plt.grid(True)

# DB 指数图
plt.subplot(1, 3, 3)
plt.plot(n_clusters_range, db_scores, marker='o')
plt.title('Davies-Bouldin 指数确定最优簇数（越小越好）')
plt.xlabel('簇数量 (n_clusters)')
plt.ylabel('DB 指数')
plt.grid(True)

plt.tight_layout()
plt.show()

# 提示用户选择 n_clusters 值（这里可以根据图表选择最佳簇数）
selected_n_clusters = 5 # 示例值，根据图表调整

# 使用选择的簇数进行 Agglomerative Clustering 聚类
agglo = AgglomerativeClustering(n_clusters=selected_n_clusters, linkage='ward')
agglo_labels = agglo.fit_predict(X_scaled)
X['Agglo_Cluster'] = agglo_labels

# 使用 PCA 降维到 2D 进行可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Agglomerative Clustering 聚类结果可视化
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agglo_labels, palette='viridis')
plt.title(f'Agglomerative Clustering with n_clusters={selected_n_clusters} (PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 打印 Agglomerative Clustering 聚类标签的分布
print(f"Agglomerative Cluster labels (n_clusters={selected_n_clusters}) added to X:")
print(X[['Agglo_Cluster']].value_counts())

# 层次聚类的树状图可视化
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# 假设 X_scaled 是标准化后的数据
# 计算层次聚类的链接矩阵
Z = hierarchy.linkage(X_scaled, method='ward')  # 'ward' 是常用的合并准则

# 绘制树状图
plt.figure(figsize=(10, 6))
hierarchy.dendrogram(Z, truncate_mode='level', p=3)  # p 控制显示的层次深度
# hierarchy.dendrogram(Z, truncate_mode='level')  # 不用p这个参数，可以显示全部的深度
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()