#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)
movies.shape


# In[4]:


credits.head(1)['cast']


# In[5]:


credits.head(1)['cast'].values


# In[13]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head()


# In[8]:


movies.head(1)


# In[9]:


movies['original_language'].value_counts()


# In[10]:


movies.info()


# In[14]:


## columns to keep (based on whether it will help to create tags)
# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id_x','genres','keywords','title','cast_x','crew_x','overview']]


# In[15]:


movies.head()


# In[16]:


## dealing with missing data

movies.isnull().sum()


# In[17]:


movies.dropna(inplace=True)


# In[18]:


movies.isnull().sum()


# In[19]:


movies.duplicated().sum()


# In[20]:


movies.drop_duplicates(inplace=True)


# In[21]:


movies.duplicated().sum()


# In[22]:


movies.iloc[0].genres


# In[23]:


## preprocessing (change format of '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[24]:


# def convert(obj):
#     L = []
#     for i in obj:
#         L.append(i['name'])
#     return L                  ## helper function


# In[25]:


# ## above list '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ## is string of list so we convert to list  (to do this we have module in python 'ast' and 'literal_eval' function under ast to do this task)

# import ast
# ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[26]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L                  ## helper function


# In[27]:


# movies['genres'].apply(convert)


# In[28]:


movies['genres'] = movies['genres'].apply(convert)


# In[29]:


movies.head()


# In[31]:


## now for keywords

movies['keywords'] = movies['keywords'].apply(convert)


# movies.head()

# In[32]:


movies.head()


# In[35]:


movies['cast_x'][0]


# In[36]:


## now we taking name of 1st 3 cast of the movie

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L                 


# In[37]:


movies['cast_x'].apply(convert3)


# In[38]:


movies['cast_x'] = movies['cast_x'].apply(convert3)


# In[39]:


movies.head()


# In[40]:


movies['crew_x'][0]


# In[41]:


## to fetch director from crew

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L                  


# In[43]:


movies['crew_x'].apply(fetch_director)


# In[44]:


movies['crew_x'] = movies['crew_x'].apply(fetch_director)


# In[45]:


movies.head()


# In[46]:


movies['overview'][0]


# In[47]:


## overview is in string format
## we goint to convert it in list so that we can concatenate with other list

movies['overview'].apply(lambda x:x.split())


# In[48]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[49]:


movies.head()


# In[50]:


## he problem might encounter in movie recommender system when there is space inbetween 

movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


# In[52]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast_x'] = movies['cast_x'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew_x'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


# movies.head()

# In[53]:


movies.head()


# In[58]:


## we will create tag and concatenate above columns

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast_x'] + movies['crew_x']


# In[59]:


movies.head()


# In[60]:


## new dataframe

new_df = movies[['movie_id_x','title','tags']]


# In[61]:


new_df


# In[62]:


## we now convert list of tag into string

new_df['tags'].apply(lambda x:" ".join(x))


# In[63]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[64]:


new_df.head()


# In[66]:


new_df['tags'][0]


# In[67]:


## convert to lower case

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[ ]:





# In[109]:


new_df['tags'][0]


# In[68]:


new_df.head()


# In[95]:


import nltk


# In[111]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[113]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[118]:


new_df['tags']=new_df['tags'].apply(stem)


# In[119]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[ ]:





# In[140]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[141]:


vectors


# In[142]:


vectors[0]


# In[144]:


## to see most commonly used words

cv.get_feature_names_out()


# In[145]:


len(cv.get_feature_names_out())


# In[146]:


['loved','loving','love']
['love','love','love']


# In[147]:


ps.stem('love')


# In[148]:


ps.stem('dance')


# In[149]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver action adventure fantasy sciencefiction')


# In[150]:


from sklearn.metrics.pairwise import cosine_similarity


# In[151]:


similarity = cosine_similarity(vectors)


# In[152]:


similarity[1]


# In[154]:





# In[157]:


new_df[new_df['title'] =="Avatar"].index[0]


# In[158]:


new_df[new_df['title'] =="Batman Begins"].index[0]


# In[159]:


list(enumerate(similarity[0]))


# In[161]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[172]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[173]:


recommend('Avatar')


# In[171]:


new_df.iloc[61].title


# In[175]:


recommend('Batman Begins')


# In[176]:


import pickle


# In[177]:


pickle.dump(new_df,open('movies.pkl','wb'))        ## wb= write binary mode


# In[178]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))        ## wb= write binary mode


# In[179]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




