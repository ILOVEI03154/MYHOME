# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# # 加载鸢尾花数据集
# iris = load_iris()
# X = iris.data
# y = iris.target

# # PCA降维
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# # 可视化
# plt.figure(figsize=(12, 5))

# # PCA可视化
# plt.subplot(1, 2, 1)
# for label in np.unique(y):
#     plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=iris.target_names[label])
# plt.title("PCA Visualization")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()

# # t-SNE可视化
# plt.subplot(1, 2, 2)
# for label in np.unique(y):
#     plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=iris.target_names[label])
# plt.title("t-SNE Visualization")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.legend()

# plt.tight_layout()
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 读取 data.csv 文件
data = pd.read_csv('python60-days-challenge\\python-learning-library\\data.csv')

# 检查数据是否存在缺失值
print("数据缺失值情况：")
print(data.isnull().sum())

# 假设最后一列为标签，其余列为特征
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# 对标签进行编码
le = LabelEncoder()
y = le.fit_transform(y)

# 识别分类特征和数值特征
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

# 打印识别的特征类型，用于检查
print("分类特征：", categorical_features)
print("数值特征：", numerical_features)

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 预处理数据
X_transformed = preprocessor.fit_transform(X)

# 计算最大允许的 n_components
n_features = X_transformed.shape[1]
n_classes = len(np.unique(y))
max_n_components = min(n_features, n_classes - 1)

# SVD 降维
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_transformed)

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed)

# t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_transformed)

# LDA 降维
lda = LinearDiscriminantAnalysis(n_components=max_n_components)
X_lda = lda.fit_transform(X_transformed, y)

# SVD 可视化
plt.figure(figsize=(8, 8))
for label in np.unique(y):
    plt.scatter(X_svd[y == label, 0], X_svd[y == label, 1], label=le.inverse_transform([label])[0])
plt.title("SVD Visualization")
plt.xlabel("SVD 1")
plt.ylabel("SVD 2")
plt.legend()
plt.tight_layout()
plt.show()

# PCA 可视化
plt.figure(figsize=(8, 8))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=le.inverse_transform([label])[0])
plt.title("PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

# t-SNE 可视化
plt.figure(figsize=(8, 8))
for label in np.unique(y):
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=le.inverse_transform([label])[0])
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.tight_layout()
plt.show()

# LDA 可视化
plt.figure(figsize=(8, 8))
if X_lda.shape[1] == 1:
    for label in np.unique(y):
        plt.scatter(X_lda[y == label, 0], np.zeros_like(X_lda[y == label, 0]), label=le.inverse_transform([label])[0])
    plt.ylabel('')
    plt.yticks([])
else:
    for label in np.unique(y):
        plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=le.inverse_transform([label])[0])
    plt.ylabel("LDA 2")
plt.title("LDA Visualization")
plt.xlabel("LDA 1")
plt.legend()
plt.tight_layout()
plt.show()

