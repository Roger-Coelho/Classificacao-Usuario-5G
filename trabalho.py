import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


#Importando dados da base de 5G do Kaggle:
df_sample = pd.read_csv('sample.csv')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_x = pd.read_csv('complete_base.csv')
print(df_train.shape, df_test.shape, df_sample.shape)

#Exploração de dados
df_train.head()
df_sample.head(2)
df_train.info()
print(df_train.head())
print(df_sample.head(2))
print(df_train.info())

df_train.prov_id.value_counts()
print(df_train.prov_id.value_counts().count()," Categorias")

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

#Não nulos
false_columns=[]
for i in df_test.columns:
    if i not in df_train.columns:
        false_columns.append(i)

assert false_columns == []

#Visualização de dados
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

#Área de interesse
df_train_area_of_interest = ['user_id','prov_id', 'chnl_type', 'service_type', 'product_type','activity_type','sex','manu_name', 'term_type', 'max_rat_flag', 'is_5g_base_cover','is_work_5g_cover', 'is_home_5g_cover', 'is_work_5g_cover_l01','is_home_5g_cover_l01', 'is_work_5g_cover_l02', 'is_home_5g_cover_l02','is_act_expire', 'comp_type', 'city_5g_ratio', 'city_level', 'is_5g']
df_test_area_of_interest = df_train_area_of_interest.copy()
df_complete_area_of_interest = df_train_area_of_interest.copy()
df_test_area_of_interest.remove('is_5g')

#Previsão de dados
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

#Conjunto de teste 1
#Modelo Logistic Regression
print("Modelo Logistic Regression")
model1 = LogisticRegression(solver = 'liblinear')
model1.fit(x_train, y_train)
pred = model1.predict(x_test)
print(classification_report(y_test, pred))

#Modelo de Árvore de Decisão
print("Modelo de Árvore de Decisão")
model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
pred2 = model2.predict(x_test)
print(classification_report(y_test, pred2))

#Random Forest
print("Modelo Random Forest")
model3 = RandomForestClassifier()
model3.fit(x_train, y_train)
pred3 = model3.predict(x_test)
print(classification_report(y_test, pred3))

#XGBoost
print("Modelo XGBoost")
model4 = XGBClassifier()
model4.fit(x_train, y_train)
pred4 = model4.predict(x_test)
print(classification_report(y_test, pred4))

#Modelo Naive Bayes
print("Modelo Naive Bayes")
model5 = GaussianNB()
model5.fit(x_train, y_train)
pred5 = model5.predict(x_test)
print(classification_report(y_test, pred5))

#Modelo K-NN
print("Modelo K-NN")
model6 = KNeighborsClassifier(n_neighbors = 50)
model6.fit(x_train, y_train)
pred6 = model6.predict(x_test)
print(classification_report(y_test, pred6))


#Conjuto de Teste 2
#df_x = juntar a base de 1 milhão
x = df_x.drop(columns = ['user_id', 'is_5g'])
y = df_x['is_5g']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.3, random_state = 42)

#Modelo Logistic Regression
print("Modelo Logistic Regression")
model1 = LogisticRegression(solver = 'liblinear')
model1.fit(x_train, y_train)
pred_test1 = model1.predict(x_test)
print(classification_report(y_test, pred_test1))

#Modelo de Árvore de Decisão
print("Modelo de Árvore de Decisão")
model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
pred_test2 = model2.predict(x_test)
print(classification_report(y_test, pred_test2))

#Random Florest
print("Modelo Random Forest")
model3 = RandomForestClassifier()
model3.fit(x_train, y_train)
pred_test3 = model3.predict(x_test)
print(classification_report(y_test, pred_test3))

#XGBoost
print("Modelo XGBoost")
model4 = XGBClassifier()
model4.fit(x_train, y_train)
pred_test4 = model4.predict(x_test)
print(classification_report(y_test, pred_test4))

#Modelo Naive Bayes
print("Modelo Naive Bayes")
model5 = GaussianNB()
model5.fit(x_train, y_train)
pred_test5 = model5.predict(x_test)
print(classification_report(y_test, pred_test5))

#Modelo K-NN
print("Modelo K-NN")
model6 = KNeighborsClassifier(n_neighbors = 50)
model6.fit(x_train, y_train)
pred_test6 = model6.predict(x_test)
print(classification_report(y_test, pred_test6))


#Comparação de Modelos
model_compare = pd.DataFrame(list(zip(y_test, pred_test1, pred_test2, pred_test3, pred_test4, pred_test5, pred_test6)), columns = ['ActualSet', 'LogisticRegression', 'DecisionTree', 'RandomForest', 'XGB', 'NaiveBayes', 'K-NN'])
population = model_compare.shape[0]

elements1, count = \
np.unique(np.where(model_compare['LogisticRegression'] == model_compare['ActualSet'],True,False),return_counts=True)
print('Contagem de Corretos e Falsos: Logistic Regression')
print('Classificações corretas: {} - Classificações falsas: {}'.format(count[1], count[0]))
correct1 = count[1] * 100 / population

elements2, count = \
np.unique(np.where(model_compare['DecisionTree'] == model_compare['ActualSet'], True, False), return_counts = True)
print('Contagem de Corretos e Falsos: Decision Tree')
print('Classificações corretas: {} - Classificações falsas: {}'.format(count[1], count[0]))
correct2 = count[1] * 100 / population

elements3, count = \
np.unique(np.where(model_compare['RandomForest'] == model_compare['ActualSet'], True, False), return_counts = True)
print('Contagem de Corretos e Falsos: Random Forest')
print('Classificações corretas: {} - Classificações falsas: {}'.format(count[1], count[0]))
correct3 = count[1] * 100 / population

elements4, count = \
np.unique(np.where(model_compare['XGB'] == model_compare['ActualSet'], True, False), return_counts = True)
print('Contagem de Corretos e Falsos: XGBoost')
print('Classificações corretas: {} - Classificações falsas: {}'.format(count[1], count[0]))
correct4 = count[1] * 100 / population

elements5, count = \
np.unique(np.where(model_compare['NaiveBayes'] == model_compare['ActualSet'], True, False), return_counts = True)
print('Contagem de Corretos e Falsos: Naive Bayes')
print('Classificações corretas: {} - Classificações falsas: {}'.format(count[1], count[0]))
correct5 = count[1] * 100 / population

elements6, count = \
np.unique(np.where(model_compare['K-NN'] == model_compare['ActualSet'], True, False), return_counts = True)
print('Contagem de Corretos e Falsos: K-NN')
print('Classificações corretas: {} - Classificações falsas: {}'.format(count[1], count[0]))
correct6 = count[1] * 100 / population
#print('Classificações corretas: {} - Classificações falsas: {}'.format(count[0],count[0]-model_compare['KNN'].shape[0]))
#correct6 = count[0] * 100 / population


print("")
print("Porcentagem de acertos dos Classificadores")
print("Porcentagem de acertos Logistic Regression: ", correct1)
print("Porcentagem de acertos Decision Tree: ", correct2)
print("Porcentagem de acertos Random Forest: ", correct3)
print("Porcentagem de acertos XGBoost: ", correct4)
print("Porcentagem de acertos Naive Bayes: ", correct5)
print("Porcentagem de acertos K-NN: ", correct6)
print("")

#Plotagem do Gráfico Comparativo
acc = {'Logistic Regression' : correct1, 'Decision Tree' : correct2, 'Random Forest' : correct3, 'XGBoost' : correct4, 'Naive Bayes' : correct5, 'K-NN' : correct6}
acc = sorted(acc.items(), key = lambda x: x[1], reverse = True)
acc = dict(acc)

plt.figure(figsize = [25, 10])
plt.plot(acc.keys(), acc.values())
plt.ylabel('Porcentagem de Precisão - Acurácia', fontsize = 30)
plt.xlabel('Classificadores de Aprendizagem de Máquina', fontsize = 30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.title('Classificação de Usuário 5G',fontsize = 50)
plt.grid()
plt.tight_layout()
plt.show()

#Saída do CSV com os comparativos
model_compare.to_csv('comparativos.csv', index = False)