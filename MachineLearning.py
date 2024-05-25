##########################
# Sales Prediction with Linear Regression
##########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x : "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

###########################
# Simple Linear Regression with OLS Using Scikit_Learn
###########################

df = pd.read_csv("Datasets/advertising.csv")

df.shape

X = df[["TV"]]
y = df[["sales"]]

###############################
# Model
################################

reg_model = LinearRegression().fit(X ,y)

# y_hat = b + w*TV

#sabit (b - bias)
reg_model.intercept_[0]
#tv'nin katsayısı (w1)
reg_model.coef_[0][0]

#Tahmin
#150 birimlik tv harcaması olsa ne kadar satış olması beklenir ?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

#500 birimlik tv harcaması olsa ne kadar satış olması beklenir ?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T

#Modelin Görselleştirilmesi

g = sns.regplot(x=X, y=y , scatter_kws={"color":"b","s":9},ci = False , color = "r" )
g.set_title(f"Model Denklemi : Sales = {round(reg_model.intercept_[0],2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10 , 310)
plt.ylim(bottom = 0)
plt.show()

#######################
# Tahmin Başarısı
#######################

#MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean()
y.std()
#10.51

#RMSE
np.sqrt(mean_squared_error(y, y_pred))
#3.24

#MAE
mean_absolute_error(y , y_pred)
#2.54

#R -KARE Veri setindeki bağımsız değişkenlerin bağımlı değişkenleri açıklama yüzdesidir.
reg_model.score(X,y)

# Çoklu Doğrusal Regresyon Modeli

df = pd.read_csv("Datasets/advertising.csv")

X = df.drop("sales", axis = 1)

y = df[["sales"]]

##########################
#Model
###########################

X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size = 0.20 , random_state = 1 )

X_train.shape
y_train.shape

X_test.shape
y_test.shape

reg_model = LinearRegression().fit(X_train , y_train)

#sabit (b - bias)
reg_model.intercept_

#coefficients (w - weights)
reg_model.coef_

#Tahmin
#Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir ?
# TV : 30
# radio : 10
# newspaper : 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

# Sales = 2.90  +   TV * 0.04 + radio + 0.17 + newspaper * 0.002

2.90 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri =  [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

# Çoklu Doğrusal Regresyonda Tahmin Başarısı

# Tahmin Başarısını Değerlendirme

#Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train , y_pred))
#1.73

#Train RKARE
reg_model.score(X_train , y_train)

#Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test , y_pred))
#1.41

#Test RKARE
reg_model.score(X_test , y_test)

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv =10,scoring = "neg_mean_squared_error" )))

#1.69

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv =5,scoring = "neg_mean_squared_error" )))
#1.71  

########################################
# Simple Linear Regression with Gradient Descent from Scratch
########################################

#Cost Function MSE
def cost_function(Y,b,w,X):
    m = len(Y)
    sse = 0

    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m
    return mse

#update_weights
def update_weights(Y , b, w, X , learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b , new_w


######################
# Lojistik Regresyon
######################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score , confusion_matrix , classification_report ,RocCurveDisplay
from sklearn.model_selection import train_test_split , cross_validate


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

pd.set_option("display.max_columns" , None)
pd.set_option("display.float_format" , lambda x : "%.3f" % x )
pd.set_option("display.width" , 500)

#Exploratory Data Analysis

df = pd.read_csv("Datasets/diabetes.csv")
df.head()
df.shape

##################
# Target'ın Analizi
##################

df["Outcome"].value_counts()

sns.countplot(x ="Outcome" , data = df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

######################
#Feature 'ların Analizi
#######################

df.describe().T # Sadece sayısal değişkneleri getirir. Ve onların durumunu özetlemektedir.


df["BloodPressure"].hist(bins =  20)
plt.xlabel("BloodPressure")
plt.show()

df["Glucose"].hist(bins =  20)
plt.xlabel("Glucose")
plt.show()

def plot_numerical_col (dataframe , numerical_col):
    dataframe[numerical_col].hist(bins =  20)
    plt.xlabel(numerical_col)
    plt.show(block = True)

for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

for col in cols :
    plot_numerical_col(df , col)

########################
# Target vs Features
########################

df.groupby("Outcome").agg({"Pregnancies":"mean"})

def target_summary_with_num(dataframe , target , numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end = "\n\n\n")

for col in cols :
    target_summary_with_num(df, "Outcome",col)

################################
#Data Preprocessing (Veri Ön İşleme )
##################################

df.shape
df.head()

df.isnull().sum()

df.describe().T

for col in cols :
    print(col , check_outlier(df, col))

replace_with_thresholds(df ,"Insulin")

#Standartlaşma 2 açıdan önem taşımaktadır.
#1.Modellerin değişkenlere eşit yaklaşması gerekmektedir.
#2.Kullanılan parametre tahmin  yöntemlerinin daha hızlı ve daha doğru tahminlerde bulunması içindir.

for col in cols :
    df[col] = RobustScaler().fit_transform(df[[col]])
#RobustScaler'ın StandartScaler'a göre farkı aykırı değerlerden etkilenmemesidir.

##################################
# Model & Prediction
##################################

y = df["Outcome"]

X = df.drop(["Outcome"] , axis = 1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred [0:10]

y[0:10]

#################################
# Model Evaluation
#################################

def plot_confusion_matrix(y , y_pred):
    acc = round(accuracy_score(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm , annot = True , fmt = ".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score : {0}".format(acc), size = 10)
    plt.show()

plot_confusion_matrix(y , y_pred)

print(classification_report(y , y_pred))

#Accuracy : 0.78
#Precision : 0.74
#Recall : 0.58
# F1-score : 0.65

#ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y , y_prob)
#0.83939

#######################
# Model Validation : Holdout
########################

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.20,random_state=17)

log_model  = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1] # 1 sınıfına ait olma olasılığı

print(classification_report(y_test , y_pred))

#Accuracy : 0.77
#Precision : 0.79
#Recall : 0.53
# F1-score : 0.63

#AUC
roc_auc_score(y_test,y_prob)

############################################
#Model Validation : 10-Fold Cross Validation
############################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X,y)

cv_results = cross_validate(log_model,
                            X,y,
                            cv = 5,
                            scoring=["accuracy","precision","recall","f1","roc_auc"])

cv_results["test_accuracy"].mean()

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)