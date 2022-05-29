import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Carregamento de Dados
df = pd.read_csv('./heart.csv')
df.sample(3)

X = df.drop('target', axis=1)
Y = df['target']

# Normalização de Dados
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Redução da Dimensionalidade
pca = PCA(n_components=2).fit(X)
X_reduced = pca.fit_transform(X)

# Novos dados
df_2d = pd.DataFrame(X_reduced)
df_2d.columns = ['0', '1']

# Visualização por classe
sns.scatterplot(data=df_2d, x='0', y='1', hue=Y, style=Y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=26)
print('\n\ntreinamento:', len(y_train))
print('teste      :', len(y_test))


# --------------------------------------------------------------------- #
# K-NN (Nearest Neighbors)

model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
acc = accuracy_score(y_pred, y_test)
pre = precision_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test)

print('\n\nK-NN')
print('\nAcurácia K-NN:', int(acc * 100), '%')
print('Precisão K-NN:', int(pre * 100), '%')
print('F1-Score K-NN:', int(f1 * 100), '%')


# --------------------------------------------------------------------- #
# SVM (Support Vector Machines)
model2 = SVC()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

acc_svm = accuracy_score(y_pred, y_test)
pre_svm = precision_score(y_pred, y_test)
f1_svm = f1_score(y_pred, y_test)

print('\n\nSVM')
print('\nAcurácia SVM:', int(acc_svm * 100), '%')
print('Precisão SVM:', int(pre_svm * 100), '%')
print('F1-Score SVM:', int(f1_svm * 100), '%')


# --------------------------------------------------------------------- #
# Random Forest
model3 = RandomForestClassifier(n_estimators=100, random_state=26)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
acc_random = accuracy_score(y_pred, y_test)
pre_random = precision_score(y_pred, y_test)
f1_random = f1_score(y_pred, y_test)

print('\n\nRandom Forest')
print('\nAcurácia Random Forest:', int(acc_random * 100), '%')
print('Precisão Random Forest:', int(pre_random * 100), '%')
print('F1-Score Random Forest:', int(f1_random * 100), '%')
