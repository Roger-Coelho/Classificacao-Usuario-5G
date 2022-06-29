import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Importing Data On Kaggle:
df_sample = pd.read_csv('sample.csv')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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

plt.figure(figsize = [25, 10])
plt.subplot(2, 1, 1)
b = sb.distplot(df_train[df_train['is_5g'] == 1]['city_5g_ratio'], color = "skyblue")
b.set_xlabel("city_5g_ratio de is_5g = 1", fontsize = 25)
b.set_ylabel("Densidade", fontsize = 25)
b.tick_params(labelsize = 20)
plt.subplot(2, 1, 2)
b = sb.distplot(df_train[df_train['is_5g'] == 0]['city_5g_ratio'], color = "red")
b.set_xlabel("city_5g_ratio de is_5g = 0", fontsize = 25)
b.set_ylabel("Densidade", fontsize = 25)
b.tick_params(labelsize = 20)
plt.tight_layout()
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "prov_id", y = "is_5g", data = df_train)
b.axes.set_title("investigação prov_id com is_5g", fontsize = 50)
b.set_xlabel("prov_id", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "chnl_type", y = "is_5g", hue = 'service_type', data = df_train)
b.axes.set_title("Tipo de canal vs is_5g no tipo de serviço", fontsize = 50)
b.set_xlabel("Tipos de Canal", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20')
plt.setp(b.get_legend().get_title(), fontsize = '25')
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "chnl_type", y = "is_5g", hue = 'product_type', data = df_train)
b.axes.set_title("Tipo de canal vs is_5g no tipo de produto", fontsize = 50)
b.set_xlabel("Tipos de Canal", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "product_type", y = "is_5g", hue = 'service_type', data = df_train)
b.axes.set_title("Tipo de produto vs is_5g no tipo de serviço", fontsize = 50)
b.set_xlabel("Tipos de Produto", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "activity_type", y = "is_5g", hue = 'sex', data = df_train)
b.axes.set_title("Tipo de atividade vs is_5g no tipo de sexo", fontsize = 50)
b.set_xlabel("Tipos de Atividades", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "comp_type", y = "is_5g", hue = 'is_act_expire', data = df_train)
b.axes.set_title("Tipo de Comp v.s is_5g em is_act_expire", fontsize = 50)
b.set_xlabel("Tipo de Comp", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "term_type", y = "is_5g", hue = 'max_rat_flag', data = df_train)
b.axes.set_title("Tipo de termo v.s is_5g em max_rat_flag", fontsize = 50)
b.set_xlabel("Tipo de Termo", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "age", y = "is_5g", hue = 'sex', data = df_train)
b.axes.set_title("Idade vs is_5g em sexo", fontsize = 50)
b.set_xlabel("Idade", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "city_level", y = "is_5g", hue = 'sex', data = df_train)
b.axes.set_title("Nível da cidade vs is_5g no tipo de sexo", fontsize = 50)
b.set_xlabel("Nível da Cidade", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()

plt.figure(figsize = [25, 10])
b = sb.barplot(x = "manu_name", y = "is_5g", hue = 'sex', data = df_train)
b.axes.set_title("manu_name v.s is_5g no tipo de sexo", fontsize = 50)
b.set_xlabel("Manu Name", fontsize = 30)
b.set_ylabel("is_5g", fontsize = 30)
b.tick_params(labelsize = 20)
plt.setp(b.get_legend().get_texts(), fontsize = '20') # for legend text
plt.setp(b.get_legend().get_title(), fontsize = '25') # for legend title
plt.show()