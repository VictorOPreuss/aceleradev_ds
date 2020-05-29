#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


# Function to calculate % of values in a column

def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# In[4]:


black_friday.head(5)


# In[5]:


#black_friday.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)


# In[6]:


black_friday.info()


# In[7]:


# Number of unique classes in each DTYPE column

black_friday.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[8]:


percent_value_counts(black_friday, 'Age')


# In[9]:


percent_value_counts(black_friday, 'City_Category')


# In[10]:


percent_value_counts(black_friday, 'Stay_In_Current_City_Years')


# In[11]:


# Number of unique classes in each DTYPE column

black_friday.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)


# In[12]:


percent_value_counts(black_friday, 'Occupation')


# In[13]:


# Number of unique classes in each DTYPE column

black_friday.select_dtypes('float64').apply(pd.Series.nunique, axis = 0)


# In[14]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[15]:


missing_values = missing_values_table(black_friday)
missing_values.head(25)


# In[16]:


black_friday.shape


# In[17]:


black_friday.describe()


# In[18]:


filt = black_friday[(black_friday.Age == '26-35') & (black_friday.Gender == 'F')].User_ID.unique()
len(filt)


# In[19]:


users = black_friday.User_ID.unique()
users


# In[20]:


types = black_friday.dtypes.nunique()
types


# In[21]:


non_null_values = len(black_friday.dropna(axis=0, how='any'))
null_values = len(black_friday) - non_null_values
percent_null = null_values/len(black_friday)
percent_null


# In[22]:


black_friday.isna().sum().max()


# In[23]:


black_friday['Product_Category_3'].value_counts().idxmax()


# In[24]:


mean = black_friday.Purchase.mean()
std = black_friday.Purchase.std()
Purchase_norm = (black_friday.Purchase - mean)/(std)


# In[25]:


Purchase_norm.mean()


# In[26]:


Purchase_norm.between(-1,1).sum()


# In[27]:


df = black_friday[black_friday['Product_Category_2'].isna() == True]
df2 = black_friday.iloc[df.index,:].Product_Category_3.isna()
df2.all()


# In[28]:


df.index


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[29]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[30]:


def q2():
    # Retorne aqui o resultado da questão 2.
    filt = black_friday[(black_friday.Age == '26-35') & (black_friday.Gender == 'F')]
    return len(filt)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[31]:


def q3():
    # Retorne aqui o resultado da questão 3.
    users = black_friday.User_ID.nunique()
    return users


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[32]:


def q4():
    # Retorne aqui o resultado da questão 4.
    types = black_friday.dtypes.nunique()
    return types


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[33]:


def q5():
    # Retorne aqui o resultado da questão 5.
    non_null_values = len(black_friday.dropna(axis=0, how='any'))
    null_values = len(black_friday) - non_null_values
    percent_null = null_values/len(black_friday)
    return percent_null


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[34]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isna().sum().max().item() 


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[35]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[44]:


def q8():
    # Retorne aqui o resultado da questão 8.
    min = black_friday.Purchase.min()
    max = black_friday.Purchase.max()
    Purchase_norm = (black_friday.Purchase - min)/(max - min)
    return Purchase_norm.mean().item()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[37]:


def q9():
    # Retorne aqui o resultado da questão 9.
    mean = black_friday.Purchase.mean()
    std = black_friday.Purchase.std()
    Purchase_norm = (black_friday.Purchase - mean)/(std)
    return Purchase_norm.between(-1,1).sum().item() 


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[38]:


def q10():
    # Retorne aqui o resultado da questão 10.
    df = black_friday[black_friday['Product_Category_2'].isna() == True]
    df2 = black_friday.iloc[df.index,:].Product_Category_3.isna()
    return df2.all().item()

