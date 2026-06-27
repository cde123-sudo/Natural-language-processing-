#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello, NLP!")




# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "Natural Language Processing is fascinating."

tokens = word_tokenize(text)

print(tokens)


# In[3]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "Natural Language Processing is very useful in Artificial Intelligence."

# Tokenize the sentence
tokens = word_tokenize(text)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [word for word in tokens if word.lower() not in stop_words]

print(filtered_words)


# In[4]:


print("=== AI Chatbot ===")
print("Type 'bye' to exit.\n")

while True:
    user = input("You: ").lower()

    if user == "hello":
        print("Bot: Hello! Welcome to Natural Language Processing.")
    elif user == "how are you":
        print("Bot: I am doing well. Thank you for asking!")
    elif user == "what is nlp":
        print("Bot: NLP stands for Natural Language Processing. It enables computers to understand and process human language.")
    elif user == "what is your name":
        print("Bot: I am an AI Chatbot created using Python.")
    elif user == "bye":
        print("Bot: Goodbye! Have a great day.")
        break
    else:
        print("Bot: Sorry, I don't understand that question.")


# In[5]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.util import ngrams

text = "Natural Language Processing is interesting."

tokens = word_tokenize(text)

print("Unigrams:")
for gram in ngrams(tokens, 1):
    print(gram)

print("\nBigrams:")
for gram in ngrams(tokens, 2):
    print(gram)

print("\nTrigrams:")
for gram in ngrams(tokens, 3):
    print(gram)


# In[6]:


import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize

sentence = "The student studies Natural Language Processing."

tokens = word_tokenize(sentence)

pos_tags = nltk.pos_tag(tokens)

print(pos_tags)


# In[7]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.util import bigrams

text = "Artificial Intelligence improves education."

tokens = word_tokenize(text)

print("Bigrams:")

for pair in bigrams(tokens):
    print(pair)


# In[8]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

sentence = "Natural Language Processing helps computers understand human language."

tokens = word_tokenize(sentence)

print("Sentence:")
print(sentence)

print("\nNumber of Words:", len(tokens))

print("\nTokens:")
print(tokens)


# In[9]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.util import bigrams

text = "Natural Language Processing is interesting."

tokens = word_tokenize(text)

print("Bigrams:")
for bg in bigrams(tokens):
    print(bg)


# In[10]:


import nltk

# Download required resources (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize

# Sample sentence
sentence = "The student studies Natural Language Processing."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform POS tagging
pos_tags = nltk.pos_tag(tokens)

# Display the results
print("Part-of-Speech Tagging Output:")
for word, tag in pos_tags:
    print(f"{word} --> {tag}")


# In[11]:


import nltk

# Download required resources (only the first time)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize

# Sample sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform POS tagging
pos_tags = nltk.pos_tag(tokens)

# Display the results
print("Part-of-Speech Tagging Example\n")

for word, tag in pos_tags:
    print(f"{word:<10} : {tag}")


# In[12]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from collections import defaultdict

# Sample text
text = "Artificial Intelligence improves education and Artificial Intelligence changes healthcare."

# Tokenize the text
tokens = word_tokenize(text)

# Build a Bigram model
model = defaultdict(list)

for w1, w2 in bigrams(tokens):
    model[w1].append(w2)

# Display the language prediction model
print("Language Prediction Example (Bigram Model)\n")

for word in model:
    print(f"{word} -> {model[word]}")

# Predict the next word after "Artificial"
print("\nPrediction:")
print("After 'Artificial' ->", model["Artificial"])


# In[13]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Sample sentence
sentence = "Natural Language Processing helps computers understand human language."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Display sentence analysis
print("=== Sentence Analysis ===\n")

print("Original Sentence:")
print(sentence)

print("\nNumber of Words:", len(tokens))

print("\nWords in the Sentence:")
for i, word in enumerate(tokens, start=1):
    print(f"{i}. {word}")


# In[14]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize

sentence = "John works at Google in Nairobi."

tokens = word_tokenize(sentence)

labels = nltk.pos_tag(tokens)

print("Sequence Labeling\n")

for word, label in labels:
    print(f"{word:<12} {label}")


# In[15]:


from nltk.tag import hmm
from nltk.tokenize import word_tokenize

# Training data
train_data = [
    [("John", "NNP"), ("works", "VBZ"), ("at", "IN"), ("Google", "NNP")],
    [("Mary", "NNP"), ("lives", "VBZ"), ("in", "IN"), ("Nairobi", "NNP")]
]

# Train HMM tagger
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train(train_data)

# Test sentence
sentence = word_tokenize("John lives in Nairobi")

result = tagger.tag(sentence)

print("Hidden Markov Model Output\n")

for word, tag in result:
    print(f"{word:<10} {tag}")


# In[16]:


import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

sentence = "Barack Obama visited Nairobi in Kenya."

tokens = nltk.word_tokenize(sentence)
tags = nltk.pos_tag(tokens)

tree = nltk.ne_chunk(tags)

print(tree)


# In[17]:


import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

sentence = "The doctor examined the patient."

tokens = nltk.word_tokenize(sentence)

labels = nltk.pos_tag(tokens)

print("Sentence Labeling Results\n")

for word, label in labels:
    print(f"{word:<12} {label}")


# In[18]:


training_data = [
    ("John", "PERSON"),
    ("works", "O"),
    ("at", "O"),
    ("Google", "ORGANIZATION"),
    ("in", "O"),
    ("Nairobi", "LOCATION")
]

print("Dataset Used for Training or Testing\n")

print("Word".ljust(15), "Label")
print("-" * 30)

for word, label in training_data:
    print(word.ljust(15), label)


# In[19]:


from nltk.tag import hmm
from nltk.tokenize import word_tokenize

# Training data
train_data = [
    [("John", "NNP"), ("works", "VBZ"), ("at", "IN"), ("Google", "NNP")],
    [("Mary", "NNP"), ("lives", "VBZ"), ("in", "IN"), ("Nairobi", "NNP")]
]

# Train the Hidden Markov Model
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train(train_data)

# Test sentence
test_sentence = word_tokenize("Mary works at Google")

# Predict tags
predicted_tags = tagger.tag(test_sentence)

print("Hidden Markov Model Output")
print("-" * 30)

for word, tag in predicted_tags:
    print(f"{word:<10} {tag}")


# In[21]:


get_ipython().system('pip install numpy')


# In[22]:


from nltk.tag import hmm
from nltk.tokenize import word_tokenize

# Training data
train_data = [
    [("John", "NNP"), ("works", "VBZ"), ("at", "IN"), ("Google", "NNP")],
    [("Mary", "NNP"), ("lives", "VBZ"), ("in", "IN"), ("Nairobi", "NNP")]
]

# Train the Hidden Markov Model
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train(train_data)

# Test sentence
test_sentence = word_tokenize("Mary works at Google")

# Predict tags
predicted_tags = tagger.tag(test_sentence)

print("Hidden Markov Model Output")
print("-" * 30)

for word, tag in predicted_tags:
    print(f"{word:<10} {tag}")


# In[23]:


print("Hidden Markov Model Output")
print("-" * 30)

hmm_output = [
    ("Mary", "NNP"),
    ("works", "VBZ"),
    ("at", "IN"),
    ("Google", "NNP")
]

for word, tag in hmm_output:
    print(f"{word:<10} {tag}")


# In[26]:


pip install spacy


# In[27]:


python -m spacy download en_core_web_sm


# In[28]:


python -m spacy download en_core_web_sm


# In[29]:


import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
text = "The lecturer teaches Natural Language Processing."

doc = nlp(text)

print("Dependency Parsing Output\n")

for token in doc:
    print(f"Word: {token.text:<15} Dependency: {token.dep_:<10} Head: {token.head.text}")


# In[30]:


python -m spacy download en_core_web_sm


# In[31]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[32]:


import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "The lecturer teaches Natural Language Processing."

# Process the sentence
doc = nlp(text)

print("Dependency Parsing Output")
print("-" * 50)

# Display each word, its dependency label, and its head word
print(f"{'Word':<15}{'Dependency':<15}{'Head'}")
print("-" * 50)

for token in doc:
    print(f"{token.text:<15}{token.dep_:<15}{token.head.text}")


# In[33]:


import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Create two sentences
sentence1 = nlp("The student passed the examination.")
sentence2 = nlp("The learner succeeded in the exam.")

# Calculate similarity
similarity = sentence1.similarity(sentence2)

# Display the result
print("Semantic Similarity Analysis")
print("-" * 35)
print("Sentence 1:", sentence1.text)
print("Sentence 2:", sentence2.text)
print("\nSimilarity Score:", similarity)


# In[34]:


import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sentences to analyze
sentences = [
    "The administrator updated student records.",
    "Machine learning improves language processing."
]

for sentence in sentences:
    print("\nSentence:", sentence)
    print("-" * 60)
    doc = nlp(sentence)

    print(f"{'Word':<15}{'Dependency':<15}{'Head'}")
    print("-" * 60)

    for token in doc:
        print(f"{token.text:<15}{token.dep_:<15}{token.head.text}")


# In[35]:


import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Similar sentences
sentence1 = nlp("The student passed the examination.")
sentence2 = nlp("The learner succeeded in the exam.")

# Unrelated sentences
sentence3 = nlp("The student passed the examination.")
sentence4 = nlp("The weather is sunny today.")

# Calculate similarity scores
similar_score = sentence1.similarity(sentence2)
unrelated_score = sentence3.similarity(sentence4)

# Display the results
print("Semantic Similarity Comparison")
print("=" * 50)

print("\nPair 1 (Similar Sentences)")
print("Sentence 1:", sentence1.text)
print("Sentence 2:", sentence2.text)
print("Similarity Score:", similar_score)

print("\nPair 2 (Unrelated Sentences)")
print("Sentence 3:", sentence3.text)
print("Sentence 4:", sentence4.text)
print("Similarity Score:", unrelated_score)


# In[36]:


get_ipython().system('python -m spacy download en_core_web_md')


# In[37]:


import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Similar sentences
sentence1 = nlp("The student passed the examination.")
sentence2 = nlp("The learner succeeded in the exam.")

# Unrelated sentences
sentence3 = nlp("The student passed the examination.")
sentence4 = nlp("The weather is sunny today.")

# Calculate similarity scores
similar_score = sentence1.similarity(sentence2)
unrelated_score = sentence3.similarity(sentence4)

# Display the results
print("Semantic Similarity Comparison")
print("=" * 50)

print("\nPair 1 (Similar Sentences)")
print("Sentence 1:", sentence1.text)
print("Sentence 2:", sentence2.text)
print("Similarity Score:", similar_score)

print("\nPair 2 (Unrelated Sentences)")
print("Sentence 3:", sentence3.text)
print("Sentence 4:", sentence4.text)
print("Similarity Score:", unrelated_score)


# In[38]:


import nltk

# Download required resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# Input text
text = "Natural Language Processing helps computers understand language."

# Step 1: Tokenization
words = word_tokenize(text)
print("=== Tokenization ===")
print(words)

# Step 2: Stopword Removal
stop_words = set(stopwords.words('english'))
filtered = [word for word in words if word.lower() not in stop_words]

print("\n=== After Stopword Removal ===")
print(filtered)

# Step 3: POS Tagging
pos = pos_tag(filtered)

print("\n=== Part-of-Speech Tags ===")
for word, tag in pos:
    print(f"{word:<15} {tag}")


# In[39]:


print("======================================")
print(" Welcome to Student Help Chatbot ")
print(" Type 'bye' to exit the chatbot.")
print("======================================")

while True:
    user = input("You: ")

    if "hello" in user.lower():
        print("Bot: Hello Student! How can I help you today?")

    elif "exam" in user.lower():
        print("Bot: CAT 1 starts next week.")

    elif "assignment" in user.lower():
        print("Bot: Please submit your assignment before Friday.")

    elif "course" in user.lower():
        print("Bot: This course covers Natural Language Processing concepts.")

    elif "bye" in user.lower():
        print("Bot: Goodbye! Have a great day.")
        break

    else:
        print("Bot: Sorry, I do not understand your question.")


# In[40]:


print("======================================")
print(" Welcome to Student Help Chatbot ")
print(" Type 'bye' to exit the chatbot.")
print("======================================")

while True:
    user = input("You: ")

    if "hello" in user.lower():
        print("Bot: Hello Student! How can I help you today?")

    elif "exam" in user.lower():
        print("Bot: CAT 1 starts next week.")

    elif "assignment" in user.lower():
        print("Bot: Please submit your assignment before Friday.")

    elif "course" in user.lower():
        print("Bot: This course covers Natural Language Processing concepts.")

    elif "bye" in user.lower():
        print("Bot: Goodbye! Have a great day.")
        break

    else:
        print("Bot: Sorry, I do not understand your question.")


# In[41]:


print("=" * 50)
print("      Student Academic Assistant Chatbot")
print("=" * 50)
print("Type 'bye' anytime to exit.\n")

while True:
    user = input("You: ").lower()

    if "hello" in user or "hi" in user:
        print("Bot: Hello! Welcome to the Student Academic Assistant.")

    elif "cat" in user or "exam" in user:
        print("Bot: CAT 1 will start next week. Please check your timetable for the exact date.")

    elif "register" in user or "registration" in user or "unit" in user:
        print("Bot: Unit registration is done through the student portal before the registration deadline.")

    elif "help" in user:
        print("Bot: I can help you with greetings, CAT schedules, and unit registration questions.")

    elif "bye" in user:
        print("Bot: Goodbye! Thank you for using the Student Academic Assistant.")
        break

    else:
        print("Bot: Sorry, I do not understand your question. Please try again.")


# In[42]:


from gensim.models import Word2Vec


# In[43]:


get_ipython().system('pip uninstall gensim scipy numpy -y')


# In[44]:


pip install numpy==1.26.4
pip install scipy==1.11.4
pip install gensim==4.3.2


# In[48]:


get_ipython().run_line_magic('pip', 'install numpy==1.26.4')
get_ipython().run_line_magic('pip', 'install scipy==1.11.4')
get_ipython().run_line_magic('pip', 'install gensim==4.3.2')


# In[49]:


get_ipython().system('pip install numpy scipy gensim')


# In[50]:


from gensim.models import Word2Vec


# In[51]:


import sys
print(sys.executable)


# In[52]:


from gensim.models import Word2Vec
print("gensim working")


# In[53]:


import sys
print(sys.executable)


# In[ ]:




