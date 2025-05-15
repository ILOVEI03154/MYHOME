import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'python60-days-challenge\python-learning-library\train.csv')
print(data.head())
print(data.info())

#有缺失值
print(data.isnull().sum())

#补全缺失值
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

#删除不需要的列
data.drop([ 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

print(data.isnull().sum())



#数据预处理
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

#划分训练集和测试集
from sklearn.model_selection import train_test_split
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#训练模型
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#模型评估
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
#预测
# 添加 PassengerId 特征
new_data = pd.DataFrame({'PassengerId': [3], 'Pclass': [3], 'Age': [45], 'SibSp': [1], 'Parch': [0], 'Fare': [10], 'Sex_male': [1], 'Embarked_Q': [1], 'Embarked_S': [0]})
new_data_pred = model.predict(new_data)
print('Prediction:', new_data_pred)

#保存模型
import joblib
joblib.dump(model, 'titanic_model.pkl')

#对测试集进行预测
test_data = pd.read_csv(r'python60-days-challenge\python-learning-library\test.csv')
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# 删除不需要的列
test_data.drop(['Name', 'Ticket'], axis=1, inplace=True)

# 数据预处理
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# 确保测试数据的特征与训练数据一致
missing_cols = set(X_train.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[X_train.columns]

x_test = test_data

model = joblib.load('titanic_model.pkl')    
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns=['Survived'])
result = pd.concat([test_data['PassengerId'], y_pred], axis=1)
result.to_csv('titanic_submission.csv', index=False)
print(result.head())



# 基础版代码（准确率约0.78）
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 数据加载
train = pd.read_csv(r'C:\Users\I.Love.I\Desktop\Python_code\python60-days-challenge\python-learning-library\train.csv')
test = pd.read_csv(r'C:\Users\I.Love.I\Desktop\Python_code\python60-days-challenge\python-learning-library\test.csv')

# 数据预处理
def preprocess(df):
    # 从姓名中提取头衔
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 填充缺失值：根据头衔填充年龄
    title_age_mean = df.groupby('Title')['Age'].mean()
    for title in title_age_mean.index:
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = title_age_mean[title]
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # 特征工程
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 0, 'IsAlone'] = 1
    
    # 头衔编码
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    return df

train = preprocess(train)
test = preprocess(test)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# 模型调优：使用网格搜索寻找最优参数
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 模型评估
from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"交叉验证平均准确率: {scores.mean():.4f}")

# 生成提交文件
predictions = best_model.predict(X_test)
output = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
output.to_csv('submission_optimized.csv', index=False)


