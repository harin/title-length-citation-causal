
# Introduction

Recently, I was working on a [project](https://github.com/harin/dblp-data-exploration) for Exploratory Data Analysis and Visualization class, where we have chosen to analyze the trends and relationship between computer science paper from the DBLP database. One finding we found is that there is a negative correlation between title length and the number of citations. Since I recently took the Causal Inference class, this is a good opportunity to apply them and see whether shorter title length in fact causes more citations.

Note that the data was taken from https://aminer.org/citation.


```python
import pandas as pd
import numpy as np
from causality.estimation.parametric import PropensityScoreMatching
import statsmodels.api as sm
from graphviz import Digraph
%matplotlib inline
```


```python
df = pd.read_pickle('dblp.pkl')
df['title_len_words'] = df.title.apply(lambda x: len(x.split(' ')))
df.sample(100000).plot('title_len_words', 'n_citation', kind='scatter');
```


![png](output_2_0.png)


To investigate whether title length causes number of citation to increase, we first need to come up with a causal graph we believe represents the system. One of the simplest graph we can have is one in which the title length effects the number of citation, but is confounded by the author. This is represented by the graph below:


```python
g = Digraph()
g.node('D', 'Title length')
g.node('Z', 'Author')
g.node('Y', 'Citations')
g.edges(['DY', 'ZD', 'ZY'])
g
```




![svg](output_4_0.svg)




```python
df.shape
```




    (3079007, 8)



There is two problems with our causal variable "Title length" is that it is not a binary variable, this is a departure from what I've learned, so the result might not be correct. Second, "Author" is a categorical data, so we have to convert them to dummy variables, which would limit how many authors we can include in our analysis.

## Data Mangling


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>id</th>
      <th>n_citation</th>
      <th>references</th>
      <th>title</th>
      <th>venue</th>
      <th>year</th>
      <th>title_len_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Makoto Satoh, Ryo Muramatsu, Mizue Kayama, Ka...</td>
      <td>00127ee2-cb05-48ce-bc49-9de556b93346</td>
      <td>0</td>
      <td>[51c7e02e-f5ed-431a-8cf5-f761f266d4be, 69b625b...</td>
      <td>Preliminary Design of a Network Protocol Learn...</td>
      <td>international conference on human-computer int...</td>
      <td>2013</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Gareth Beale, Graeme Earl]</td>
      <td>001c58d3-26ad-46b3-ab3a-c1e557d16821</td>
      <td>50</td>
      <td>[10482dd3-4642-4193-842f-85f3b70fcf65, 3133714...</td>
      <td>A methodology for the physically accurate visu...</td>
      <td>visual analytics science and technology</td>
      <td>2011</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Altaf Hossain, Faisal Zaman, Mohammed Nasser,...</td>
      <td>001c8744-73c4-4b04-9364-22d31a10dbf1</td>
      <td>50</td>
      <td>[2d84c0f2-e656-4ce7-b018-90eda1c132fe, a083a1b...</td>
      <td>Comparison of GARCH, Neural Network and Suppor...</td>
      <td>pattern recognition and machine intelligence</td>
      <td>2009</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Jea-Bum Park, Byungmok Kim, Jian Shen, Sun-Yo...</td>
      <td>00338203-9eb3-40c5-9f31-cbac73a519ec</td>
      <td>0</td>
      <td>[8c78e4b0-632b-4293-b491-85b1976675e6, 9cdc54f...</td>
      <td>Development of Remote Monitoring and Control D...</td>
      <td></td>
      <td>2011</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Giovanna Guerrini, Isabella Merlo]</td>
      <td>0040b022-1472-4f70-a753-74832df65266</td>
      <td>2</td>
      <td>NaN</td>
      <td>Reasonig about Set-Oriented Methods in Object ...</td>
      <td></td>
      <td>1998</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (3079007, 8)




```python
is_list = df.authors.apply(lambda x: isinstance(x, list))
```


```python
df[~is_list]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>id</th>
      <th>n_citation</th>
      <th>references</th>
      <th>title</th>
      <th>venue</th>
      <th>year</th>
      <th>title_len_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1105528</th>
      <td>NaN</td>
      <td>634390c1-d4cf-4192-b55f-8ade3af72a08</td>
      <td>0</td>
      <td>[11e73009-08b3-4361-bd8d-89312b7cc7fa, 866260c...</td>
      <td>Elastogram estimation using adaptive-length Sa...</td>
      <td>biomedical engineering and informatics</td>
      <td>2011</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1174531</th>
      <td>NaN</td>
      <td>736b86b6-715c-4b3b-8d38-d6cd03709bb2</td>
      <td>1</td>
      <td>NaN</td>
      <td>Pose Invariant Face Recognition by Face Synthe...</td>
      <td>british machine vision conference</td>
      <td>2000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2594452</th>
      <td>NaN</td>
      <td>9c4cf6a4-3d7a-4892-9acd-dc30336c73f1</td>
      <td>1</td>
      <td>[1a6ecea3-bad3-4139-8c15-9a35247b8be4, 93cffd7...</td>
      <td>An efficient intra-mode decision method for HEVC</td>
      <td>Signal, Image and Video Processing</td>
      <td>2016</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2881547</th>
      <td>NaN</td>
      <td>81d297e4-0b71-4f14-81e6-7fe51abd1d31</td>
      <td>3</td>
      <td>NaN</td>
      <td>Durchgängiges modellbasiertes Engineering von ...</td>
      <td>Automatisierungstechnik</td>
      <td>2016</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove unused columns and remove rows without authors
df = df[['authors', 'n_citation', 'title_len_words']] 
df = df[is_list]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>n_citation</th>
      <th>title_len_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Makoto Satoh, Ryo Muramatsu, Mizue Kayama, Ka...</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Gareth Beale, Graeme Earl]</td>
      <td>50</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Altaf Hossain, Faisal Zaman, Mohammed Nasser,...</td>
      <td>50</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Jea-Bum Park, Byungmok Kim, Jian Shen, Sun-Yo...</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Giovanna Guerrini, Isabella Merlo]</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['num_authors'] = df.authors.apply(len)
df.num_authors.sum()
```




    9476165




```python
df = df.reset_index()
```


```python
# unroll the authors column
rows = []
for _, row in df[is_list].iterrows():
    for author in row.authors:
        rows.append([author, row['index']])
```

    C:\Users\harinsa\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      This is separate from the ipykernel package so we can avoid doing imports until
    


```python
author_df = pd.DataFrame(rows)
```


```python
author_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Makoto Satoh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ryo Muramatsu</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mizue Kayama</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kazunori Itoh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Masami Hashimoto</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
author_df.columns = ['author', 'index']
author_df = author_df.set_index('index') 
```


```python
author_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Makoto Satoh</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Ryo Muramatsu</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Mizue Kayama</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Kazunori Itoh</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Masami Hashimoto</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = author_df.join(df)[['author', 'title_len_words', 'n_citation', 'num_authors']].reset_index()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>author</th>
      <th>title_len_words</th>
      <th>n_citation</th>
      <th>num_authors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Makoto Satoh</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ryo Muramatsu</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Mizue Kayama</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Kazunori Itoh</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Masami Hashimoto</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert author to integers
df.author = df.author.astype('category')
df.author = df.author.cat.rename_categories(np.arange(len(df.author.cat.categories)))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>author</th>
      <th>title_len_words</th>
      <th>n_citation</th>
      <th>num_authors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>967189</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1344877</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1091158</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>831977</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1021921</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
paper_count = df[['author', 'n_citation']].groupby('author').count()
```


```python
# filter out author with only 1 paper, not sure if this is necessary because we are not doing any matching, but
# intuitively, it is probably better to have author with paper with multiple paper length to regress on
valid_authors = paper_count[paper_count.n_citation > 1]
```


```python
# use only author with more than one paper
df = df[df.author.isin(valid_authors.index)]
```


```python
# remove rows without data
df = df.dropna()
```


```python
df.shape
```




    (8577232, 5)




```python
df.author.cat.categories
```




    Int64Index([      0,       1,       2,       3,       4,       5,       6,
                      7,       8,       9,
                ...
                1766536, 1766537, 1766538, 1766539, 1766540, 1766541, 1766542,
                1766543, 1766544, 1766545],
               dtype='int64', length=1766546)



Still over a million author, would be impossible to do matching. Let's perform matching on a sample of authors instead.


```python
def regress(n):
    author_sample = df.author.cat.categories.to_series().sample(n)
    df_sample = df[df.author.isin(author_sample)]
    df_sample.author = df_sample.author.astype(int).astype('category')
    df_sample.author = df_sample.author.cat.rename_categories(np.arange(len(df_sample.author.cat.categories)))

    X = pd.concat([pd.get_dummies(df_sample.author, prefix='author'), df_sample['title_len_words']], axis=1)
    sm.add_constant(X)
    y = df_sample['n_citation']
    model = sm.OLS(y, X)

    fit = model.fit()
    return fit
```


```python
regress(100).conf_int().loc['title_len_words']
```

    C:\Users\harinsa\Anaconda3\lib\site-packages\pandas\core\generic.py:4401: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value
    




    0   -2.130604
    1    0.512479
    Name: title_len_words, dtype: float64




```python
regress(1000).conf_int().loc['title_len_words']
```

    C:\Users\harinsa\Anaconda3\lib\site-packages\pandas\core\generic.py:4401: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value
    




    0   -3.208329
    1    0.215463
    Name: title_len_words, dtype: float64



# Conclusion

If OLS is a valid way to remove confounding between "Title length" and "Number of Citations" then, this experiment have not quite proven that shorter title length results in more citation, as the confidence interval of the coefficient of the title length may still be due to randomness. 

More important, we first need to verify whether OLS is in fact a good estimator for the causal effect in presence of *non-binary causal state* and *categorical confounder*. Furthermore, here we are not nearly close to the amount of data we have. Each sample we run give conflicting results, some successfully rejects the null, while other did not.

