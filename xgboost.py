import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


#Importing Data On Kaggle:
df_sample = pd.read_csv('sample.csv')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_x = pd.read_csv('complete_base.csv')
print(df_train.shape, df_test.shape, df_sample.shape)

#Data Exploration
df_train.head()
df_sample.head(2)
df_train.info()
print(df_train.head())
print(df_sample.head(2))
print(df_train.info())

df_train.prov_id.value_counts()
print(df_train.prov_id.value_counts().count()," Categories")

df_train.area_id.value_counts()
print(df_train.area_id.value_counts())

df_train.user_id.value_counts()
print(df_train.user_id.value_counts())

df_train.nunique()
print(df_train.nunique())

print(df_train.active_days01)

days=['active_days01', 'active_days02',
       'active_days03', 'active_days04', 'active_days05', 'active_days06',
       'active_days07', 'active_days08', 'active_days09', 'active_days10',
       'active_days11', 'active_days12', 'active_days13', 'active_days14',
       'active_days15', 'active_days16', 'active_days17', 'active_days18',
       'active_days19', 'active_days20', 'active_days21', 'active_days22',
       'active_days23']
df_train.drop(columns=days,inplace=True)
df_test.drop(columns=days,inplace=True)

assert df_train.isnull().sum().sum()==0 , df_test.isnull().sum().sum() == 0

#No nulls
false_columns=[]
for i in df_test.columns:
    if i not in df_train.columns:
        false_columns.append(i)

assert false_columns == []

#Data Visualization
print(df_train.columns,df_train.columns.shape[0])

#Interest Area
df_train_area_of_interest = ['user_id','prov_id', 'chnl_type', 'service_type', 'product_type','activity_type','sex','manu_name', 'term_type', 'max_rat_flag', 'is_5g_base_cover','is_work_5g_cover', 'is_home_5g_cover', 'is_work_5g_cover_l01','is_home_5g_cover_l01', 'is_work_5g_cover_l02', 'is_home_5g_cover_l02','is_act_expire', 'comp_type', 'city_5g_ratio', 'city_level', 'is_5g']
df_test_area_of_interest = df_train_area_of_interest.copy()
df_complete_area_of_interest = df_train_area_of_interest.copy()
df_test_area_of_interest.remove('is_5g')

#Data Prediction
df_train = df_train[df_train_area_of_interest]
df_test = df_test[df_test_area_of_interest]
df_x = df_x[df_complete_area_of_interest]

df_train.head()
print(df_train.head())

df_test.head()
print(df_test.head())

train_user_id = df_train['user_id']
test_user_id = df_test['user_id']
x_train = df_train.drop(columns = ['user_id', 'is_5g'])
y_train = df_train['is_5g']
x_test = df_test.drop('user_id', axis = 1)
y_test = df_sample['is_5g']

#XGBoost
print("Modelo XGBoost")
model4 = XGBClassifier()
model4.fit(x_train, y_train)
pred4 = model4.predict(x_test)
print(classification_report(y_test, pred4))

#Conjunto de teste 2
#df_x = juntar a base de 1 milhão
x = df_x.drop(columns = ['user_id', 'is_5g'])
y = df_x['is_5g']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.3, random_state = 42)
model4 = XGBClassifier()
model4.fit(x_train, y_train)
pred_test4 = model4.predict(x_test)
print(classification_report(y_test, pred_test4))