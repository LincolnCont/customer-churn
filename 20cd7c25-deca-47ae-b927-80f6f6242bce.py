#!/usr/bin/env python
# coding: utf-8

# # Модель предсказания оттока клиентов телекоммуникационной компании

# Оператор связи «ТелеДом» хочет бороться с оттоком клиентов. Для этого его сотрудники начнут предлагать промокоды и специальные условия всем, кто планирует отказаться от услуг связи. Чтобы заранее находить таких пользователей, «ТелеДому» нужна модель, которая будет предсказывать, разорвёт ли абонент договор. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и услугах. Ваша задача — обучить на этих данных модель для прогноза оттока клиентов.

# ## Импорт файлов и библиотек

# In[1]:


get_ipython().system('wget https://code.s3.yandex.net/data-scientist/ds-plus-final.db ')


# In[2]:


get_ipython().system('pip install -q phik')


# In[3]:


import math
from math import ceil 
from pathlib import Path 
import random 
import re 
import warnings 
warnings.filterwarnings('ignore')
from datetime import datetime
import time

from catboost import CatBoostClassifier
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:,.4f}'.format)
from scipy import stats
import seaborn as sns
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.metrics import average_precision_score

import phik
from phik.report import plot_correlation_matrix
from phik import report

import sqlalchemy
from sqlalchemy import create_engine 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential,Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
RANDOM_STATE = 160824


# In[4]:


import os
import pandas as pd
from sqlalchemy import create_engine


path_to_db = 'ds-plus-final.db'
engine = create_engine(f'sqlite:///{path_to_db}', echo=False) 


# In[5]:


engine.table_names()


# In[6]:


query = '''
SELECT *
FROM contract
'''

df_contract = pd.read_sql_query(query, con=engine)
df_contract.head()


# In[7]:


query = '''
SELECT *
FROM personal
'''

df_personal = pd.read_sql_query(query, con=engine)
df_personal.head()


# In[8]:


query = '''
SELECT *
FROM internet
'''

df_internet = pd.read_sql_query(query, con=engine)
df_internet.head()


# In[9]:


query = '''
SELECT *
FROM phone
'''

df_phone = pd.read_sql_query(query, con=engine)
df_phone.head()


# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Данные успешно загружены.
# </div>

# ## Предобработка и исследовательский анализ данных

# Напишем функцию для отображении информации о датасете

# In[10]:


def df_info(df, figsize=(27, 9)):
    print('Shape:')
    print(df.shape)
    print('Head:')
    display(df.head())
    print('Describe df:')
    display(df.describe())
    print(); print()
    print(df.info())
    print(); print()
    print('Nan values:')
    display(df.isna().mean())
    print(); print()
    print('Unique values:')
    print(df.nunique())
    print(); print()
    print('Full duplicates:')
    print(df.duplicated().sum())
    print(); print()
    print('Distributions of numerical values:')
    try:
        df.hist(figsize=figsize)
    except:
        print('No numericals found')


# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Молодец, что оформляешь код в виде функций для переиспользования.
# </div>

# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 1</b>: Чтобы на экран не выводилось непонятное None, не нужно вызов info оборачивать в display или print.
# </div>

# ### Таблица 'contract'

# In[11]:


df_info(df_contract)


# Заменим типа данных у 'BeginDate' и 'EndDate' 

# In[12]:


df_contract['BeginDate'] = pd.to_datetime(df_contract['BeginDate'], format='%Y-%m-%d')
df_contract['BeginDate'].head()


# In[13]:


df_contract.head(30)


# In[14]:


df_contract['MonthlyCharges'] = df_contract['MonthlyCharges'].replace(' ','')


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Для чего нужна строка кода выше?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Без этой строки не удавалось изменить тип данных
# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 2</b>: В исследовательском проекте такую аномалию лучше задокументировать и вывести примеры данных, чтобы можно было довести информацию до поставщика данных.
# </div>
#     
# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 2</b>: Мне казалось, что пробелы есть не в MonthlyCharge, а в TotalCharges.
# </div>
# </div>

# In[15]:


df_contract['MonthlyCharges'] = df_contract['MonthlyCharges'].astype(float)
df_contract.loc[df_contract['TotalCharges'].isna(), 'TotalCharges'] = df_contract['MonthlyCharges']


# Заменим пропуски в TotalСharges на MonthlyCharges, так как пропуски у новых абонентов

# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Из чего следует, что пропуски у новых абонентов? Нужно привести срезы данных для обоснования выводов.</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Ниже вывел строки с отсутствующими значениями TotalCharges и у всех начала контракта совпадает с датой формирования базы данных
# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 2</b>: Теперь наглядно видно, что клиенты с неявными пропусками в TotalCharges, действительно являются новыми.
# </div>
# </div>

# In[16]:



df_contract['TotalCharges'] = df_contract['TotalCharges'].fillna(df_contract['MonthlyCharges'])


# In[17]:


df_contract['TotalCharges'] = df_contract['TotalCharges'].replace(' ', np.nan)


df_contract['TotalCharges'] = pd.to_numeric(df_contract['TotalCharges'], errors='coerce')

df_contract['TotalCharges'] = df_contract['TotalCharges'].fillna(df_contract['MonthlyCharges'])


# <div class="alert alert-block alert-warning">
# ⚠️ <s><b>Ревью 1</b>: Команда выше не приводит к заполнению пропусков (NaN) в TotalCharges, так как пропуски там неявные в виде пробелов (' '). Явные пропуски появятся в TotalCharges после выполнения следующей команды. Их и нужно заполнить значениями из MonthlyCharges потом.</s>
# </div>

# In[18]:


df_info(df_contract)


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Какие выводы можно сделать по представленным выше данным и визуализациям?</s>
# </div>

# Выводы:
# - В датасете представлены данные о 7043 клиентах компании, дубликаты отсутствуют
# - Изменили типы данных на верные
# - Убрали пропуски в TotalCharges, хоть их было и немного
# 
# По графикам видим, что большинство абонентов платит чуть меньше 30 денежных единиц в месяц, а общие траты у большинства клиентов около 1000 денежных единиц. 

# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 2</b>: А ещё форма распределения у некоторых числовых признаков, далекая от нормального, намекает на необходимость применить PowerTransformer.
# </div>

# ### Таблица 'personal'

# In[19]:


df_info(df_personal)


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Вижу в них Yes и No, но не булевые значения. Что имеется в виду?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Я сначала изменил булевые значения на 1 и 0, но в дальнейшем отказался от этого, но забыл удалить сообщения. Сейчас везеде удалю
# </div>

# <div class="alert alert-block alert-warning">
# ⚠️ <s><b>Ревью 1</b>: Я бы не рекомендовал ручное преобразование категориальных признаков. Лучше перед обучением моделей поручить это кодировщикам из sklearn и лучше в конвейере.</s>
# </div>

# In[20]:




df_personal.head()


# ### Таблица 'internet'

# In[21]:


df_info(df_internet)


# Размер таблицы internet меньше на 1525 объектов, чем contract и pesonal. Hе все пользуются интернет-услугами

# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Какие булевые значения?</s>
# </div>

# ### Таблица 'phone'

# In[22]:


df_info(df_phone)


# Размер опять не совпадает с предыдущему таблицами. В этот раз чуть больше чем internet, но менише чем остальные

# In[23]:




df_phone.head()


# In[24]:


df_phone = df_phone.rename(columns = {'CustomerId':'customerID'})


# ### Объединение Датафреймов

# In[25]:


df = df_contract.merge(df_personal, on='customerID')
df = df.merge(df_internet, how='left', on='customerID')
df = df.merge(df_phone, how='left', on='customerID')


# In[26]:


df.head(5)


# In[27]:


df_info(df)


# Так как количество пользователей совпадает с размером датафрейма, можем удалить customerID

# In[28]:


df = df.drop('customerID', axis=1)


# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 1</b>: Обычно идентификатор не удаляют, а переносят в индекс, чтобы потом можно было вернуть прогноз модели и сопоставить его пользователю из базы данных.
# </div>

# In[29]:


print('Пропущенные значения в датафрейме:')

df.isna().sum()


# Отсутствие услуг по подключению к интернету говорит о том что человек не использует интернет компании. В связи с этим можно заgолнить пропуски поля датафрейма df_internet одним значением: '-1'. Отсутствие значение в поле 'MultipleLines' говорит о том, что у человека нет телефона. Данные пропуски также заполним значением '-1'.

# In[30]:


df = df.fillna('-1')


# <div class="alert alert-block alert-warning">
# ⚠️ <s><b>Ревью 2:</b> Код в ячейке выше приведет к заполнению пропусков в TotalCharges строкой '-1' и поменяет тип столбца с числового на строковый, что вызовет ошибки ниже по блокноту.</s>
# </div>

# <div class="alert alert-block alert-warning">
#     ⚠️ <b>Ревью 1</b>: Пропуски в категориях - действительно, признак того, что клиент не пользуется соответствующей услугой. Иногда для модели полезно отличать неиспользование услуги внутри подключенного пакета услуг (например Интернет) от услуги, не подключенной вместе с пакетом. В этом случае можно выбрать другой заполнитель, отличный от уникальных значений. Твоё решение верное по сути, но тогда и заполнитель можно использовать не строку '-1', а более понятное значение, например, 'not_connected', или 'out-of-service', или 'off'...
# </div>

# In[31]:


df.isna().sum()
df.head()


# In[32]:


def has_left(end_date):
    if end_date == 'No':
        return 0
    else:
        return 1

# Отдельное поле для данных о том, расторг ли клиент договор
df['Left'] = (df['EndDate']).apply(has_left)


# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Целевая переменная создана верно.
# </div>

# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 1</b>: Но дополнительные пояснения текстом украсили бы блокнот.
# </div>

# In[33]:


df.head()


# In[34]:


df.loc[df['EndDate'] == 'No', 'EndDate'] = pd.to_datetime('2020-02-01')

df['EndDate'] = pd.to_datetime(df['EndDate'])


df['ContractLength'] = (df['EndDate'] - df['BeginDate']).dt.days

df['ContractLength'].head()


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Что происходит в ячейке выше?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Чтобы получить нашу целевую перменную, мы заменяем пропуски в EndDate на дату форимирования БД и затем высчитываем длительность контрактов 
# </div>

# ### Исследовательский анализ данных

# In[35]:


# Функция для создания графика с долей расторгнувших договоров по категориальному признаку
def create_hist_categorical(col, rotate=0):
    sns.histplot(
        df,
        x=col,
        hue='Left',
        multiple='fill'
    )

    plt.title(f'Доля расторгнувших по "{col}"')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xticks(rotation=rotate)
    plt.ylabel('Доля')

# Функция для создания графика с долей расторгнувших договоров по количественному признаку
def create_hist_numeric(col, rotate=0):
    sns.histplot(
        df,
        x=col,
        hue='Left',
        multiple='fill',
        discrete=True
    )

    plt.title(f'Доля расторгнувших по "{col}"')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xticks(rotation=rotate)
    plt.ylabel('Доля')

# Функция для создания круговой диаграмма для бинарных признаков
def binary_pie(col, title_name):
    col_count = df.pivot_table(index=col, values='gender', aggfunc='count').reset_index()
    col_count

    col_count.loc[col_count[col] == 0, col] = 'Нет'
    col_count.loc[col_count[col] == 1, col] = 'Да'

    plt.figure(figsize=(3, 3))

    plt.pie(
        col_count['gender'],
        labels=col_count[col],
        autopct='%1.1f%%'
    )

    plt.title(title_name, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


# Зависимость пола

# In[36]:


binary_pie('Partner', 'Наличие супруга / супруги')


# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 1</b>: Распределения лучше строить в разрезе по целевой переменной после объединения таблиц, чтобы было легче увидеть "портреты" уходящего и остающегося клиентов.
# </div>

# In[37]:


binary_pie('SeniorCitizen', 'Является ли клиент пенсионером')


# In[38]:


binary_pie('PaperlessBilling', 'Наличие электронного расчетного листа')


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Какие выводы можно сделать по представленным выше данным и визуализациям?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# У меня чуть ниже идет небольшой вывод по визуализации
# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 2</b>: Я искал там упоминание расчетного листа, но его там нет. Поэтому и задаю вопрос выше.</s>
# </div>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Большинство клиентов использовали электронный расчетный лист, 
# </div>

# Диаграмма количества клиентов по объему месячных трат

# In[39]:



df.hist('MonthlyCharges', figsize=(7, 5))

plt.title('Количество клиентов по объему месячных трат', fontsize=16)
plt.xticks(fontsize=12)
plt.xlabel('Объем месячных трат', fontsize=14)
plt.ylabel('Количество клиентов', fontsize=14)
plt.show()


# Диаграмма количества клиентов по общему объему трат

# In[40]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.hist('TotalCharges', figsize=(7, 5))

plt.title('Количество клиентов по общему объему трат', fontsize=16)
plt.xticks(np.arange(0, 9001, 1000), fontsize=12)
plt.xlabel('Общий объем трат', fontsize=14)
plt.ylabel('Количество клиентов', fontsize=14)

plt.show()


# Диаграмма количества клиентов по длительности контракта в днях

# In[41]:


df.hist('ContractLength', figsize=(7, 5))

plt.title('Количество клиентов по длительности контракта, в днях', fontsize=16)
plt.xlabel('Длительность контракта в днях', fontsize=14)
plt.ylabel('Количество клиентов', fontsize=14)
plt.xticks(np.arange(0, 2500, 200), fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# По графикам видим что: 
# - людей в браке и холостых примерно поровну
# - только 16% являются пенсионерами 
# - большинство клиентов тратят ежемесячно менее 30 денежных единиц, а общие траты у большинсва клиентов меньше 1000
# - большинство клиентов имеют контракт 200 дней и менее 

# In[42]:


create_hist_categorical('gender')


# 

# In[43]:


gender_charges = df.pivot_table(index='gender', values='MonthlyCharges', aggfunc='mean').reset_index()

gender_charges.plot(
    x='gender',
    y='MonthlyCharges',
    kind='bar',
    title='Объем среднемесячных затрат по полу клиента',
    xlabel='Пол',
    ylabel='Объем среднемесячных затрат',
    legend=False,
    grid=True
)

plt.show()


# Судя по графикам количество трат и процент расторжения договора не зависит от пола клиента.

# In[44]:


create_hist_categorical('PaymentMethod', rotate=45)


# Логично что больше всего расторжений у клиентов у которых договор расторгается автоматически при неоплате в срок

# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Интересное наблюдение.
# </div>

# In[45]:


create_hist_numeric('InternetService')


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Какие выводы можно сделать по представленным выше данным и визуализациям?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Доля расторгнувших контракт выше у людей, которые использвовали fiber optic
# </div>

# In[46]:


internet_charges = df.pivot_table(index='InternetService', values='MonthlyCharges', aggfunc='mean').reset_index()

internet_charges.plot(
    x='InternetService',
    y='MonthlyCharges',
    kind='bar',
    title='Объем средних месячных трат по типу подключения',
    xlabel='Тип интернет-подключения',
    ylabel='Объем средних месячных трат',
    grid=True,
    legend=False
)

plt.show()


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Какие выводы можно сделать по представленным выше данным и визуализациям для бизнеса и машинного обучения?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Ежемесчная оплата за fiber optic на 50 процентов выше чем DSL
# </div>

# In[47]:



total_charges = df.pivot_table(index='InternetService', values='TotalCharges', aggfunc='mean').reset_index()

total_charges.plot(
    x='InternetService',
    y='TotalCharges',
    kind='bar',
    title='Объем средних общих трат по типу подключения',
    xlabel='Тип интернет-подключения',
    ylabel='Объем средних общих трат',
    grid=True,
    legend=False
)

plt.show()


# Доля расторгнувших контракт выше у людей, которые использвовали fiber optic, у них также выше объем трат на интернет( что возможно является причиной рассторжения контракта

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Объем общих трам соответвенно тоже больше в полтора раза, возможно это является причиной того, что пользователи Fiber optic на 5% процентов чаще отказываются от услуг провайдера
# </div>

# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Выше нет информации о доле расторгнувших контракт. Прошу пояснить или исправить.</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Сверху мини вывод по всем трем графикам, которые отмечены красным. Или так неверно и лучше писать под каждым?
# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 2</b>: Понятнее, когда мини-выводы следуют сразу за визуализацией, по которой они делаются.</s>
# </div>
# </div>

# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Что дает эта информация для машинного обучения и бизнеса?
# <div class="alert alert-block alert-danger">
# ❌ <b>Ревью 2</b>: Можно ли предположить, что если доли оттока у разных значений почти одинаковые, то признак, вероятно, будет менее полезен модели, чем признак в вариотивностью доли оттока?
# </div></s>
# </div>

# In[48]:


create_hist_numeric('Type')


# Очень удивило, что чаще разрывают договор люди, которые оплачивают сразу на длительный срок

# <div class="alert alert-block alert-warning">
# ⚠️ <s><b>Ревью 1</b>: Распределения лучше строить в разрезе по целевой переменной после объединения таблиц, чтобы было легче увидеть "портреты" уходящего и остающегося клиентов.</s>
# </div>

# ### Матрица корреляций 

# In[49]:


df = df.drop(['BeginDate', 'EndDate', 'MonthlyCharges'], axis=1)


# In[51]:


phik_overview = df.phik_matrix(interval_cols=['TotalCharges', 'ContractLength'])
phik_overview.round(2)

plot_correlation_matrix(
    phik_overview.values,
    x_labels=phik_overview.columns,
    y_labels=phik_overview.index,
    vmin=0,
    vmax=1,
    figsize=(15, 15)
)

plt.show()


# <div class="alert alert-block alert-danger">
# ❌ <b>Ревью 1</b>: При вызове phik_matrix в параметер interval_cols нужно передать список имен числовых признаков. Сейчас, как следует из предупреждения выше, к ним была ошибочно причислена категориальная целевая переменная 'Left'. Это может исказить корреляции, особенно в случае небинарной целевой переменной. Для бинарной расхождения будут минимальными, но нам важнее правильность методики. Нужно закрепить этот навык для будущих проектов.
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Исправил, но все равно остается целевая перменная , длина контракта и общие траты
# 
# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 3</b>: Исправления подразумевают добавление в phik_overview = df.phik_matrix() параметра interval_cols, например, так:
# 
# ```python
# phik_overview = df.phik_matrix(interval_cols=[['TotalCharges', 'ContractLength']])
# ```
#     
# Посмотри подробности в документации.</s>
# </div>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Спасибо, понял и исправил
# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 4</b>: Спасибо! Теперь нет риска расчета корреляций неподходящим для типа признака методом.
# </div>
# </div>

# По матрице видим: 
# - больше всего корреляция между параметрами интернет соединения

# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 1</b>: Если у признаков Интернет-соединения мультиколлинеарность, нужны ли они тогда все сразу в признаках для обучения моделей?
# </div>

# - самая низкая корреляция со всеми параметрами у gender 
# - целевой признак коррелируется с PaymentMethod, MonthlyCharges, total_charges, partner и  contractlength

# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Перед построением матрицы корреляций из датафрейма нужно убрать признаки, непригодные для машинного обучения, например, абсолютные даты заключения и окончания контракта. Их удаление делается не по корреляции, а так как модель не должна привязываться к датам из прошлого в признаках. Во время эксплуатации она будет видеть более свежие даты.</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Удалил
# </div>

# ## Подготовка данных к обучению

# Делаем разбивку 

# In[53]:


df['Left'].value_counts(normalize=True)


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Какие выводы можно сделать по представленным выше данным и визуализациям для бизнеса и машинного обучения?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Так как нас интересует значение ROC-AUC модели, то дисбаланс нас устраивает
# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 3</b>: Но для второй метрики из проекта (accuracy) можно учесть дисбаланс, например, гиперпараметром class_weight.
# </div>
# </div>

# In[54]:


features = df.drop('Left', axis=1)
target = df['Left']


# In[55]:


features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                            test_size = 0.25,
                                                                            stratify = target,
                                                                            random_state = RANDOM_STATE)

print(features_train.shape, target_train.shape)
print(features_test.shape, target_test.shape)


# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Разбиение на выборки сделано правильно.
# </div>

# In[56]:


categorical = [
    'Type', 'PaymentMethod', 'InternetService', 'gender', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'PaperlessBilling', 'SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines'
]
numerics = ['TotalCharges', 'ContractLength']


# In[57]:


data_transformer = ColumnTransformer(
                        transformers=[
                            ('scaler', StandardScaler(), numerics),
                            ('ohe', OneHotEncoder(sparse=False, drop='first'), categorical)],
                                )

data_transformer.fit(features_train)
features_train_transformed = data_transformer.transform(features_train)
features_test_transformed = data_transformer.transform(features_test)


# <div class="alert alert-block alert-warning">
# ⚠️ <s><b>Ревью 1</b>: Кодирование и масштабирование признаков сразу всей обучающей выборки перед кросс-валидацией является утечкой данных из валидационных фолдов в обучающие. Лучше объединять кодировщики, другие объекты преобразования признаков, включая масштабирование, и модель в конвейер для более объективной оценки модели и подбора оптимальных гиперпараметров.</s>
# </div>

# <div class="alert alert-block alert-warning">
#    
#  ⚠️ <b>Ревью 1</b>:
# 
# - Для деревянных моделей, включая бустинги над решающими деревьями, для кодирования категориальных признаков лучше подходит порядковое кодирование OrdinalEncoder, так как оно дает алгоритму обучения больше вариантов для поиска оптимального разбиения при построении дерева. У некоторых бустингов есть встронные кодировщики.
# - Для линейных моделей, KNN и нейросетей порядковые категориальные признаки можно кодировать с помощью OrdinalEncoder, а номинативные (типа цвета автомобиля) можно только с помощью OneHotEncoder или TargetEncoder.
# - Использование для каждой модели OneHotEncoder не является хорошей практикой, хотя часто встречается в сети Интернет.
# </div>

# ## Обучение моделей
# 

# ### Модель случайного леса с кросс-валидацией и подбором гиперпараметров

# In[58]:


get_ipython().run_cell_magic('time', '', 'start_time_rfc = time.time()\nparam_grid = {\n    \'n_estimators\': [10, 50, 100],\n    \'max_depth\': [10, 20, 30],\n    \'min_samples_split\': [2, 4, 6]}\n\nrfc = RandomForestClassifier(random_state=RANDOM_STATE)\ngrid_search_rfc = GridSearchCV(rfc, param_grid, scoring=\'roc_auc\', cv=StratifiedKFold(n_splits=5), verbose=1, n_jobs=-1)\ngrid_search_rfc.fit(features_train_transformed, target_train)\nprint("Лучшие параметры RandomForestClassifier: ", grid_search_rfc.best_params_)\nprint("Лучший ROC-AUC RandomForestClassifier на кросс-валидации: ", grid_search_rfc.best_score_)\n\n# Лучший ROC-AUC на кросс-валидации\nbest_roc_auc_rfc = grid_search_rfc.best_score_\nprint(f"Лучший ROC-AUC RandomForestClassifier на train: {best_roc_auc_rfc:.4f}")\n\n# Precision на кросс-валидации\nprecision_scores_rfc = cross_val_score(grid_search_rfc.best_estimator_, features_train_transformed, target_train, cv=5, scoring=\'precision\', n_jobs=-1)\nmean_precision_rfc = precision_scores_rfc.mean()\nprint(f"Среднее значение Precision RandomForestClassifier на train: {mean_precision_rfc:.4f}")\n\nend_time_rfc = time.time()\nrfc_time = end_time_rfc - start_time_rfc\nprint(f"Время выполнения RandomForestClassifier на train: {rfc_time:.4f} секунд")')


# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 1</b>: Дополнительной метрикой по заданию является не precision, а accuracy. Её можно было передать сразу списком в параметр scoring вместе с 'roc_auc', чтобы не прибегать к повторной кросс-валидации.
# </div>

# ### CatBoost с кросс-валидацией и подбором гиперпараметров

# In[59]:


get_ipython().run_cell_magic('time', '', 'start_time_cb = time.time()\nparam_grid_cb = {\n    \'iterations\': [200, 400],\n    \'depth\': [5, 10],\n    \'learning_rate\': [0.05, 0.1],\n    \'l2_leaf_reg\': [1, 3]}\n\ncb = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, cat_features=categorical)\ngrid_search_cb = GridSearchCV(cb, param_grid_cb, scoring=\'roc_auc\', cv=StratifiedKFold(n_splits=5), verbose=1, n_jobs=-1)\n# CatBoost может обрабатывать категориальные признаки напрямую\ngrid_search_cb.fit(features_train, target_train)  \n#grid_search_cb.fit(features_train_transformed, target_train)\nprint("Лучшие параметры CatBoost: ", grid_search_cb.best_params_)\nprint("Лучший ROC-AUC CatBoost на кросс-валидации: ", grid_search_cb.best_score_)\n\nbest_roc_auc_cb = grid_search_cb.best_score_\nprint(f"Лучший ROC-AUC CatBoostClassifier на train: {best_roc_auc_cb:.4f}")\n\nprecision_scores_cb = cross_val_score(grid_search_cb.best_estimator_, features_train, target_train, cv=5, scoring=\'precision\', n_jobs=-1)\nmean_precision_cb = precision_scores_cb.mean()\nprint(f"Среднее значение Precision CatBoostClassifier на train: {mean_precision_cb:.4f}")\n\nend_time_cb = time.time()\ncb_time = end_time_cb - start_time_cb\nprint(f"Время выполнения CatBoostClassifier на train: {cb_time:.4f} секунд")')


# ### Полносвязная нейронная сеть

# In[60]:


X_train = torch.FloatTensor(features_train_transformed)
X_test = torch.FloatTensor(features_test_transformed)
y_train = torch.FloatTensor(np.array(target_train)).unsqueeze(1)
y_test = torch.FloatTensor(np.array(target_test)).unsqueeze(1)
X_train.shape


# In[61]:


class CustomEarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Для чего класс выше?</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Механизм ранней остановки, чтобы модель останавливалась, когда производительность перестает улучшаться. Соответвенно модель не будет переобучаться
# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 2</b>: Такое лучше пояснять, включая критерий ранней остановки и его другие параметры, например, степень терпения.
# </div>
# </div>

# In[62]:


class Net(nn.Module):
    def __init__(self, n_in_neurons, n_hidden_neurons_1, n_hidden_neurons_2, n_out_neurons):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(n_in_neurons, n_hidden_neurons_1)
        self.bn1 = nn.BatchNorm1d(n_hidden_neurons_1)
        self.act1 = nn.ReLU()
        self.dp2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(n_hidden_neurons_1, n_hidden_neurons_2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden_neurons_2, n_out_neurons)
        self.act3 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dp2(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: В последнем слое линейный выход, который больше подходит для задачи регрессии. В этом проекте решается задача классификации. Чтобы нейросеть предсказывала вероятность, нужно на последнем слое использовать активацию сигмойдой. Для неё в качестве функции потерь подойдет BCELoss.</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Исправил
# </div>

# Определяем параметры сети

# In[63]:


n_in_neurons = X_train.shape[1]
n_hidden_neurons_1 = 24
n_hidden_neurons_2 = 10
n_out_neurons = 1

net = Net(n_in_neurons, n_hidden_neurons_1, n_hidden_neurons_2, n_out_neurons)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss = nn.BCELoss()


# Создаем DataLoader

# In[64]:


X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
dataset_train = torch.utils.data.TensorDataset(X_train_split, y_train_split)
dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
train_dataloader = DataLoader(dataset_train, batch_size=40, shuffle=True, num_workers=0)
val_dataloader = DataLoader(dataset_val, batch_size=40, num_workers=0)
test_dataloader = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=40, num_workers=0)  


# <div class="alert alert-block alert-warning">
# ⚠️ <b>Ревью 3</b>: Выделение валидационной выборки произведено после масштабирования и кодирования обучающей выборки, а это утечка из валидационной выборки в обучающую. Нужно сначала выделить валидационную из исходной обучающей выборки (которую ещё не масштабировали), потом обучить преобразователи (fit) на новой обучающей выборке, а потом применить преобразования (transform) для всех трех выборок.
# </div>

# In[65]:


get_ipython().run_cell_magic('time', '', 'early_stopping = CustomEarlyStopping(patience=5, min_delta=20)\n\nstart_time_ns = time.time()\nnum_epochs = 1000\n\nfor epoch in range(num_epochs):\n    net.train()\n    for batch in train_dataloader:\n        data_train, work_train = batch \n        optimizer.zero_grad()\n        preds = net.forward(data_train)\n        loss_value = loss(preds, work_train)\n        loss_value.backward()\n        optimizer.step()\n\n    # Оценка модели на валидационной выборке\n    if epoch % 5 == 0:\n        val_predicted_temp = [] \n        with torch.no_grad():\n            net.eval()\n            for batch in val_dataloader:\n                data_val, work_val = batch \n                val_preds = net.forward(data_val)\n                val_predicted_temp.append(val_preds)\n                BCE_loss = loss(val_preds, work_val)\n\n        val_predicted_temp = torch.cat(val_predicted_temp).detach().numpy()\n        ROC_AUC = roc_auc_score(y_val, val_predicted_temp)\n        early_stopping(BCE_loss)\n        if early_stopping.counter == 0:\n            best_roc = ROC_AUC\n            best_predicted_temp = val_predicted_temp\n        print(f"epoch:{epoch}, ROC_AUC val: {ROC_AUC}")\n\n        if early_stopping.early_stop:\n            print(\'Ранняя остановка!\')\n            print(f\'Лучший ROC-AUC на валидации: {best_roc:.4f}\')\n            break \n\n# Финальная оценка на тестовой выборке\nwith torch.no_grad():\n    test_predicted_temp = [] \n    net.eval()\n    for batch in test_dataloader:\n        data_test, work_test = batch \n        test_preds = net.forward(data_test)\n        test_predicted_temp.append(test_preds)\n\n    test_predicted_temp = torch.cat(test_predicted_temp).detach().numpy()\n    ROC_AUC_test = roc_auc_score(y_test, test_predicted_temp)\n    mean_precision_test = average_precision_score(y_test, test_predicted_temp)\n    print(f"ROC-AUC на тесте: {ROC_AUC_test:.4f}")\n    print(f"Средний Precision на тесте: {mean_precision_test:.4f}")\n\nend_time_ns = time.time()\nns_time = end_time_ns - start_time_ns\nprint(f"Время выполнения нейросети на train: {ns_time:.4f} секунд")')


# <div class="alert alert-block alert-danger">
# ❌ <s><b>Ревью 1</b>: Подбор гиперпараметров и контроль качества при обучении должен делаться не по тестовой выборке, а на кросс-валидации по обучающей выборке или отдельной валидационной выборке. Метрику на тесте нужно считать потом только для одной лучшей модели, которая будет рекомендоваться к внедрению в эксплуатацию. При обучении нейросети иногда вместо кросс-валидации по обучающей выборке используют валидационную выборку, выделяя её из обучающей. Получается три выборки: обучающая (для обучения модели), валидационная (для проверки качества от эпохи к эпохе и подбора гиперпараметров), тестовая используется только для финальной проверки одной лучшей модели, выбранной к внедрению.</s>
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> 
# Добавил валидационную выборку и на ней провели обучение 
# </div>

# ### Лучшая модель

# In[66]:


data = {'Модель': ['CatBoostClassifier', 'RandomForestClassifier', 'Полносвязная нейронная сеть'],
        'Лучший ROC-AUC': [best_roc_auc_cb, best_roc_auc_rfc, best_roc],
        'Средний Precision': [mean_precision_cb, mean_precision_rfc, mean_precision_test],
        'Время выполнения (сек)': [cb_time, rfc_time, ns_time]}

summary_table = pd.DataFrame(data)
round(summary_table, 4)


# CatBoost показал гораздо лучшие результаты, чем нейронная сеть и RandomForestClassifier

# ## Тестирование лучшей модели

# ### Проверяем качество лучшей модели на тестовой выборке

# In[67]:


best_cb = CatBoostClassifier(**grid_search_cb.best_params_, random_state=RANDOM_STATE, verbose=0, cat_features=categorical)
best_cb.fit(features_train, target_train)


# In[68]:


probs = best_cb.predict_proba(features_test)[:,1]
fpr, tpr, thresholds = roc_curve(target_test, probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.show()


# In[69]:


roc_auc_test = roc_auc_score(target_test, probs)
print(f"ROC-AUC на тестовой выборке: {roc_auc_test:.2f}")


# In[70]:


preds = best_cb.predict(features_test)
accuracy_test = accuracy_score(target_test, preds)
print(f"Accuracy на тестовой выборке: {accuracy_test:.2f}")


# ROC-AUC = 0.90 c Accuracy = 0.91 это очень хороший показатель для модели. Она имеет хорошую способность различать между положительным и отрицательным классами

# ### Матрица ошибок

# In[71]:


matrix = confusion_matrix(target_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Прогнозируемые метки')
plt.ylabel('Истинные метки')
plt.title('Матрица ошибок')
plt.show()


# По матрице имеем:
# - 1473 - количество истинно отрицательных прогнозов. Модель правильно предсказала отрицательный класс
# - 133 - количество истинно положительных прогнозов
# - 13 - количество ложноположительных прогнозов, то есть модель ошиблась
# - 142 - количество ложноотрицательных

# Отобразим полноту и точность модели

# In[72]:


recall = recall_score(target_test, preds)
precision = precision_score(target_test, preds)

labels = ['Полнота', 'Точность']
values = [recall, precision]

plt.figure(figsize=(8, 5))
bars = sns.barplot(x=labels, y=values, palette='viridis')
plt.title('Полнота и точность модели')
plt.ylabel('Оценка')
for bar in bars.patches:
    bars.annotate(f'{bar.get_height() * 100:.2f}%',
                  (bar.get_x() + bar.get_width() / 2,
                   bar.get_height()),
                   ha='center', va='center',
                   xytext=(0, 5), textcoords='offset points')

plt.show()


# Получилась очень низкая полнота, то есть не угадывает большую часть положительных случаев. Где то я намудрил, но точность осталась очень высокой 

# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Мудрения нет. Баланс между метриками всегда можно решить порогом решающего правила. Свои боли заказчики измеряют другими метриками. Порог классификации можно будет подобрать потом вместе с маркетологами, чтобы максимизировать уже не метрику машинного обучения, а метрику бизнеса в денежном выражении с учетом параметров бюджета на акции по удержанию клиентов и разной стоимости ошибок первого и второго рода.
# </div>

# ### Важность основных входных признаков

# In[73]:



importances = best_cb.get_feature_importance()

features = features_train.columns
importance_dict = {features[i]: importances[i] for i in range(len(features))}
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(15, 8))
bars = sns.barplot(x=[item[0] for item in sorted_importance], y=[item[1] for item in sorted_importance])
plt.title("Feature importances via CatBoost")
plt.ylabel('Importance')
plt.xlabel('Features')
plt.xticks(rotation=90)

for p in bars.patches:
    bars.annotate(format(p.get_height(), '.2f'),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center',
                  xytext=(0, 9),
                  textcoords='offset points')


# Что точно не так) 
# Получилось что наибольщее значения имеет длительность контракт, общие траты абонента и тип оплаты. И все бы ничего, если бы не такое огромное влияние длительности договора

# <div class="alert alert-block alert-success">
# ✔️ <b>Ревью 1</b>: Ниже ты правильно предлагаешь способ решения этого вопроса - попросить у заказчика новые более персонифицированные (пусть и обезличенные) данные о пользователях, точнее о характере пользования услугами, а не только их подключении и тарифах.
# </div>

# ## Вывод

# Проеведена работа с данными оператора связи «ТелеДом» который, хочет бороться с оттоком клиентов. Данные обработаны и подготовлены для обработки моделями. Изучали модели RandomForestClassifier, CatBoost и нейронную сеть. Лучшей моделью был CatBoost c  ROC-AUC - 0.91 и precision = 0.85. А вот у полноты(Recall) получилось низкое значение в 45%. 
# 
# Способы для дальнейшего улучшения модели
# Дополнительные данные: добавить данные.
# 
# Улучшение качества данных: устранение выбросов, обработка пропущенных значений и исправление ошибок.
# 
# Подбор гиперпараметров: осуществить поиск наилучших гиперпараметров для всех моделей.
# 
# Работа с признаками: убрать (подобрать) лишние.
# 
# Использование других моделей: XGBoost, LightGBM.
# 

# <div class="alert alert-success">
# <b>✔️ Заключение ментора:</b> Сергей, рад сообщить, что твой проект принят. Это значит, что ты справился с поставленной задачей. Поздравляю!
#     
# Спасибо за использование phik на этапе анализа данных. Очень хорошо, что ты применил в работе технику обучения моделей с подбором гиперпараметров на кросс-валидации. В новых проектах рекомендую усиливать конвейеры индивидуальным подбором методов кодирования категориальных признаков в зависимости от архитектуры модели (писал об этом в желтых сообщениях). При обработке признаков также лучше объединять конвейер предобработки и модель на кросс-валидации, чтобы сделать оценку моделей более объективной и помочь им обучаться прилежнее, устранив даже минимальный риск утечки данных из валидационных фолдов в обучающую. Здорово, что в твоем арсенале методов машинного обучения есть нейросети. Уверен, заказчику будут интересны твои предложения по улучшению его бизнеса.
# 
# Тема проекта действительно была интересная, а решение можно улучшать и дальше. Фактически наша задача является этапом более сложного процесса, называемого  <a href="https://newtechaudit.ru/vvedenie-v-uplift-modelirovanie">uplift-моделированием,</a> который подразумевает A/B тесты с контролем метрик, по результатам которых может корректироваться и порог классификации. Вот ещё некоторые источники по теме:
#     
# - [Туториал по uplift моделированию](https://habr.com/ru/companies/ru_mts/articles/485980/)
# - [Курс на ODS](https://ods.ai/tracks/uplift-modelling-course)    
# 
# Надеюсь, что финальный проект добавил мотивации к получению новых знаний. Со своей стороны хотел бы порекомендовать тебе некоторые материалы для дальнейшего совершенствования:
# 
# https://www.youtube.com/watch?v=xl1fwCza9C8 познавательное видео по настройке модели CatBoost (в ней есть много способов предотвращать переобучение);
# 
# https://habr.com/ru/company/ods/blog/322626/ на Habr можно закрепить свои знания, порешав задачи из цикла статей — Открытый курс машинного обучения;
# 
# https://github.com/esokolov/ml-course-hse — на гитхаб есть репозиторий с задачами из курса по машинному обучению от Евгения Соколова, можно использовать как дополнительный материал для закрепления знаний.
# 
# https://habr.com/ru/company/avito/blog/571094/ — материалы по A/B тестам.
# 
# Также хочу поделиться опытом и вкратце рассказать о том, в какие направления можно подаваться на работу и какие навыки там пригодятся:
# 
# Направление аналитики: хорошие знания теории вероятностей и математической статистики; базовые знания библиотек ML; уверенные знания SQL; умение решать задачи по A/B тестам; плюсом будет знание специальных инструментов для аналитики по визуализации результатов.
# 
# Направление ML: уверенные знания классических моделей машинного обучения; хорошие знания SQL; понимание алгоритмов машинного обучения — как устроена модель линейной регрессии, модели случайного леса, градиентного бустинга и т.д. На собеседовании могут спросить, как работает какая-нибудь модель «под капотом», очень любят градиентный бустинг и случайный лес. Здесь точно пригодится https://academy.yandex.ru/handbook/ml - онлайн-учебник от Школы анализа данных Яндекса, в котором описаны теоретические основы работы моделей машинного обучения.
# 
# В некоторых компаниях при устройстве на работу или стажировку, например в Яндексе, нужно решить тест на алгоритмические задачи. По алгоритмам есть много разных курсов, платных и бесплатных. Можно попробовать Тренировки по Алгоритмам от Яндекса https://yandex.ru/yaintern/algorithm-training (бесплатно, хороший курс). Также есть и у Практикума курс по алгоритмам.
#     
# Сергей, прими ещё раз мои поздравления и пожелания новых успехов!
# </div>
