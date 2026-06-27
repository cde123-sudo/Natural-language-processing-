#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


from gensim.models import Word2Vec


# In[5]:


import sys
get_ipython().system('{sys.executable} -m pip install gensim')


# In[8]:


from gensim.models import Word2Vec


# In[11]:


sentences = [
    ["i", "love", "nlp"],
    ["nlp", "is", "interesting"],
    ["machine", "learning", "is", "powerful"]
]


# In[12]:


model = Word2Vec(
    sentences,
    vector_size=50,
    window=2,
    min_count=1
)


# In[1]:


from gensim.models import Word2Vec
print("import ok")


# In[2]:


sentences = [
    ["i","love","nlp"],
    ["nlp","is","interesting"],
    ["machine","learning","is","powerful"]
]
print("data ok")


# In[3]:


model = Word2Vec(sentences, vector_size=50, window=2, min_count=1)
print("model trained")


# In[4]:


from gensim.models import Word2Vec

sentences = [
    ["i","love","nlp"],
    ["nlp","is","interesting"],
    ["machine","learning","is","powerful"]
]

model = Word2Vec(
    sentences,
    vector_size=50,
    window=2,
    min_count=1
)

print(model.wv["nlp"])


# In[5]:


print(model.wv.most_similar("nlp"))


# In[6]:


print(model.wv.similarity("nlp", "learning"))


# In[7]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[8]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[9]:


from gensim.models import Word2Vec
print("import ok")


# In[10]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[11]:


get_ipython().system('{sys.executable} -m pip install matplotlib scikit-learn')


# In[13]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[14]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
print("OK")


# In[15]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


print(result)


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[1]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[2]:


pca = PCA(n_components=2)
result = pca.fit_transform(vectors)


# In[3]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[1]:


from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[2]:


print("Hello, World!")


# In[3]:


import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()


# In[4]:


import matplotlib
print(matplotlib.__version__)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
plt.plot([1,2,3], [4,5,6])
plt.title("Test Graph")
plt.show()


# In[7]:


print(model)


# In[8]:


from gensim.models import Word2Vec


# In[9]:


sentences = [
    ["i", "love", "nlp"],
    ["nlp", "is", "interesting"],
    ["machine", "learning", "is", "powerful"]
]


# In[10]:


model = Word2Vec(
    sentences,
    vector_size=50,
    window=2,
    min_count=1
)


# In[11]:


print(model)


# In[12]:


words = list(model.wv.index_to_key)
vectors = [model.wv[word] for word in words]


# In[13]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
result = pca.fit_transform(vectors)


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

for i, word in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.text(result[i, 0] + 0.01, result[i, 1] + 0.01, word)

plt.title("Word Embeddings Visualization (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()


# In[15]:


get_ipython().system('pip install tensorflow numpy matplotlib nltk')


# In[16]:


get_ipython().system('pip install tensorflow')


# In[17]:


import sys
import platform

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Architecture:", platform.architecture())


# In[18]:


get_ipython().system('python -m pip --version')


# In[19]:


py -3.11 -m pip install tensorflow


# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install tensorflow')


# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install tensorflow')


# In[1]:


import tensorflow as tf

print(tf.__version__)


# In[2]:


import sys
import platform

print("Python Version:")
print(sys.version)

print("\nPython Executable:")
print(sys.executable)

print("\nArchitecture:")
print(platform.architecture())


# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip --version')


# In[4]:


get_ipython().system('pip install tensorflow')


# In[5]:


import sys
get_ipython().system('{sys.executable} -m pip --version')


# In[8]:


import sys
get_ipython().system('{sys.executable} -m pip config list')


# In[9]:


import sys
get_ipython().system('{sys.executable} -m pip install requests')


# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip config list')


# In[2]:


get_ipython().system('pip install tensorflow')


# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip config list')


# In[4]:


import sys
get_ipython().system('{sys.executable} -m pip install tensorflow --index-url https://pypi.org/simple --no-cache-dir')


# In[5]:


import tensorflow as tf

print(tf.__version__)


# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(8, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()


# In[7]:


from tensorflow.keras.preprocessing.text import Tokenizer

texts = [
    "I love NLP",
    "NLP is interesting"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)


# In[8]:


sequences = tokenizer.texts_to_sequences(texts)

print(sequences)


# In[9]:


import numpy as np

sample = np.array([[1,2,3,4]])

prediction = model.predict(sample)

print(prediction)


# In[10]:


from tensorflow.keras.preprocessing.text import Tokenizer

texts = [
    "I love NLP",
    "NLP is interesting",
    "Machine learning is powerful"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

print("Word Index")
print(tokenizer.word_index)

print()

print("Sequences")

print(tokenizer.texts_to_sequences(texts))


# In[12]:


get_ipython().system('pip install transformers torch')


# In[13]:


from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier(
    "I enjoyed learning Natural Language Processing."
)

print(result)


# In[14]:


from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier(
    "I enjoyed learning Natural Language Processing."
)

print(result)


# In[15]:


from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier(
    "I enjoyed learning Natural Language Processing."
)

print(result)


# In[16]:


positive = classifier(
    "This course is very interesting."
)

print(positive)


# In[17]:


negative = classifier(
    "The exam was extremely difficult."
)

print(negative)


# In[18]:


neutral = classifier(
    "Today is Monday."
)

print(neutral)


# In[19]:


from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

result = generator(
    "Artificial Intelligence will",
    max_length=50
)

print(result[0]["generated_text"])


# In[20]:


from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

result = generator(
    "Artificial Intelligence will",
    max_length=50
)

print(result[0]["generated_text"])


# In[21]:


generator(
    "The future of education is",
    max_length=40
)


# In[22]:


from transformers import pipeline

classifier = pipeline("sentiment-analysis")

texts = [
    "I love studying Artificial Intelligence.",
    "I dislike waiting in long queues.",
    "The meeting starts at 10 AM."
]

for text in texts:
    result = classifier(text)
    print("Sentence:", text)
    print("Result:", result)
    print()


# In[ ]:


from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

prompt = input("Enter a prompt: ")

result = generator(
    prompt,
    max_length=50
)

print("\nGenerated Text:\n")
print(result[0]["generated_text"])


# In[ ]:




