#########################
#Gerekli Kütüphaneler Ve Fonksiyonlar
#########################

import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x:"%.2f" % x)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

from helpers import *

df = pd.read_csv("Datasets/hitters.csv")

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols

########################################
#GELİŞMİŞ FONKSİYONEL KEŞİFCİ VERİ ANALİZİ
########################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding , Rare Encoding )
# 5. Feature Scaling (Özellik Ölçeklendirme)


##########################################
# 1. Outliers  (Aykırı Değerler)
##########################################

for col in num_cols:
    print(col , check_outlier(df,col))

for col in num_cols:
    plt.title(col)
    sns.boxplot(df[col])
    plt.show(block = True)

for col in num_cols:
    print(col , check_outlier(df,col,q1=0.1,q3=0.9))

for col in num_cols:
    if check_outlier(df,col,q1=0.1,q3=0.9):
        replace_with_thresholds(df,col,q1=0.1,q3=0.9)

###########################################
# 2.Kategorik Değişken Analizi (Analysis of Categorical Variables)
###########################################

for col in cat_cols:
    cat_summary(df,col,plot=True)

############################################
# 3.Sayısal Değişken Analizi (Analysis of Numerical Variables)
############################################

for col in num_cols:
    num_summary(df,col, plot = True)

############################################
# 4.Hedef Değişken Analizi (Analysis of Target Variable)
############################################

for col in cat_cols:
    target_summary_with_cat(df,"Salary",col)

############################################
# 5.Korelasyon Analizi (Analysis of Correlation)
############################################

high_correlated_cols(df,plot=True)

df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

############################################
# 3.Feature Extraction (Özellik Çıkarımı)
############################################

#Bağımlı değişkende boş değer varsa o doldurulmamalıdır.

cat_cols,num_cols,cat_but_car = grab_col_names(df)
num_cols

new_num_cols = [col for col in num_cols if col not in ["Salary","Years"]]

df[new_num_cols] = df[new_num_cols] + 1
#Bazı değişkenlerim sıfır olduğu için onlara bir ekliyorum.
#Çünkü aşağıda onlarla işlem yapıcam.

df.columns = [col.upper() for col in df.columns]
cat_cols,num_cols,cat_but_car = grab_col_names(df)

# RATIO OF VARIABLES

# CAREER RUNS RATIO
df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
# CAREER BAT RATIO
df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
# CAREER HITS RATIO
df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
# CAREER HMRUN RATIO
df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
# CAREER RBI RATIO
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
# CAREER WALKS RATIO
df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]
df["NEW_C_HIT_RATE"] = df["CHITS"] / df["CATBAT"]
# PLAYER TYPE : RUNNER
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHITS"]
# PLAYER TYPE : HIT AND RUN
df["NEW_C_HIT-AND-RUN"] = df["CRUNS"] / df["CHITS"]
# MOST VALUABLE HIT RATIO IN HITS
df["NEW_C_HMHITS_RATIO"] = df["CHMRUN"] / df["CHITS"]
# MOST VALUABLE HIT RATIO IN ALL SHOTS
df["NEW_C_HMATBAT_RATIO"] = df["CATBAT"] / df["CHMRUN"]

#Annual Averages
df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]


# PLAYER LEVEL
df.loc[(df["YEARS"] <= 2), "NEW_YEARS_LEVEL"] = "Junior"
df.loc[(df["YEARS"] > 2) & (df['YEARS'] <= 5), "NEW_YEARS_LEVEL"] = "Mid"
df.loc[(df["YEARS"] > 5) & (df['YEARS'] <= 10), "NEW_YEARS_LEVEL"] = "Senior"
df.loc[(df["YEARS"] > 10), "NEW_YEARS_LEVEL"] = "Expert"


# PLAYER LEVEL X DIVISION

df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Junior-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Junior-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Mid-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Mid-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Senior-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Senior-West"
df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Expert-East"
df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Expert-West"

# Player Promotion to Next League
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
df.loc[(df["LEAGUE"] == "N") & (df["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
df.loc[(df["LEAGUE"] == "A") & (df["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"



num_cols
cat_cols
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#Yeni değişken ürettiğimiz için verimiz büyüdü.Bu yüzden grab_col_names() fonksiyonuyla
#yeni değişkenleri yakalıyoruz.


# Label Encoding
#Genellikle iki değerimiz varsa  0 ve 1 şeklinde ayrı olarak kullanıyoruz.
#İkiden fazla olduğunda One-Hot Encoding kullanıyoruz.
#Ama bazı durumlarda Label Encoder'ı kullanabiliriz.
#Örneğin birbirine göre sıralama (ordinallik) farkı olan varsa Label Encoding'i kullanabilriz.

binary_cols = [col for col in df.columns if
               df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# Rare Encoding

rare_analyser(df,"SALARY", cat_cols)
df = rare_encoder(df, 0.01, cat_cols)


# 6. One-Hot Encoding
#Kategorik değişkenlerimizi numerik olarak ifade etmek için kullanırız.
#Çünkü modellerimiz ifadeleri sayısal olarak istemektedir.
# Label encoder ile farkı ise Labek encoder genellikle ordinal verilerde kullanılır.
#One-Hot Encoding ise ordinal olmayan değişkenlerde kullanılır.

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove("SALARY")

#Verilerimizi eğitmeden önce neden Scale Ederiz ?
#1.Verilerimizi aynı ölçekte görmek isteriz.(Temsili)
#2.Performans odaklı küçük sayılarda çalışmak modelleri daha rahat ettirir.

# 7. Robust-Scaler
#Aykırı değerlerden çok etkilenmemektedir.
for col in num_cols:
    transformer = RobustScaler().fit(df[col])
    df[col] = transformer.transform(df[col])

###########################################
# Multiple Linear Regression
###########################################

X= df.drop("SALARY",axis = 1)
y = df[["SALARY"]]

###############################
#Model
###############################

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train,y_train)

#b + w*i
# sabit (b - bias)
reg_model.intercept_

#coefficients (w - weights)
reg_model.coef_

#linear regression y_hat = b+w*i
np.inner(X_train.iloc[2,:].values, reg_model.coef_) + reg_model.intercept_
y_train.iloc[2]

np.inner(X_train.iloc[4,:].values, reg_model.coef_) + reg_model.intercept_
y_train.iloc[4]

################################
# Tahmin
###############################


##############################
# Tahmin Başarısını Değerlendirme
##############################

#Tahmin RMSE
#RMSE değerimiz küçüldükçe modelimiz iyi sonuçlar vermektedir.
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train , y_pred))

#Train RKARE
#Modelin açıklanabilirliğini veriyor.
#X_train verisiyle %78 bir açıklama oranına sahiptir.
reg_model.score(X_train,y_train)


#Test RMSE
#Test hatamızın train hatasından daha fazla çıkması çok normaldir.
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test , y_pred))

#Test RKARE
reg_model.score(X_test , y_test)

#10 Katlı CV RMSE
# K katlı çapraz doğrulama
#Veriyi sırayla 10 birbirinden bağımsız parçaya bölüyor.
#Böldüğü parçayı alıyor, diğer 9 parçayı eğitiyor.
np.mean(np.sqrt(-cross_val_score(reg_model
                                 ,X,y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
