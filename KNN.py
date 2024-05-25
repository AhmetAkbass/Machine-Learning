# K En Yakın Komşu (K-Nearest Neighbors)
#Gözlemlerin birbirine olan benzerlikleri üzerinden tahmin yapılır.
#KNN Regresyon ve Sınıflandırma fonksiyonları için kullanılabilir.
#Öklid ya da  benzeri bir uzaklık hesabı ile her bir gözleme uzaklık hesaplanır.
#Ardından tahmin edilmek istenen değere en yakın değerlerin ortalaması alınır.
#Sınıflandırma problmelerinde ise en yakın K adet gözlemin y değerlerinin en sık gözlenen frekansı tahmin edilen sınıf olur.

# Keşifçi Veri Analizi

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option("display.max_columns", None)


##########################################
# 1.Exploratory Data Analysis
##########################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

##########################################
# 2. Data Preprocessing & Feature Engineering
##########################################
#Uzaklık temelli yöntemlerde ve Gradient Descent'de değişkenlerin stadart olması
#daha başarılı olunmasını sağlayacaktır.
#Bu sebeple elimizdeki bağımsız değişkenleri standartlaşma işlemleri yapacağız.

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)
X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled,columns=X.columns)

##########################################
# 3.Modeling & Prediction
##########################################

knn_model =KNeighborsClassifier().fit(X,y)

random_user = X.sample(1,random_state=45)

knn_model.predict(random_user)

###########################################
# 4. Model Evaluation
###########################################

#Confusion matriz için y_pred
y_pred = knn_model.predict(X)

#AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:,1]

print(classification_report(y,y_pred))
#acc 0.83
#f1 0.74
#AUC
roc_auc_score(y,y_prob)
#0.90
# cross_validate'in cross_val_score'dan farkı birden fazla metriğe göre
#değerlendirme yapabiliyor olmasıdır.
cv_results = cross_validate(knn_model, X, y, cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#0.73
#0.59
#0.78

#Bu başarı değerleri nasıl artırılabilir?
#1. veri boyutu arttıralabilir.
#2. Veri ön işleme
#3. Özellik Mühendisliği
#4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()
#Parametre modellerin veri içinden öğrendiği
#ağırlıklardır.Özetle parametre ver içerisinden
#öğrenilmektedir.
#Hiperparametre ise kullanıcı tarafından tanımlanması gereken
#dışsal ve veri seti içerisinden öğrenilemeyen parametrelerdir.

#########################################
# 5. Hyperparameter Optimization
#########################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors" : range(2,50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X,y)
# n_jobs = -1 yapıldığında işlemcileri tam performans ile kullanır.
# verbose = yapılan işlemlerden sonra rapor bekleyip beklemediğini sorar.

knn_gs_best.best_params_

##########################################
# 6.Final Model
##########################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)
#Atama yapmak için ** ifadesini kullanırız.

cv_results = cross_validate(knn_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()