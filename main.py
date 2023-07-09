"""
This data in is in txt format with tab seperated values.
"""

import pandas as pd
import re
import nltk
# nltk.download('book')s
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

data = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])

# Data Cleaning and Preprocessing
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(len(data)):
    new = re.sub('[^a-zA-Z]',' ',data["message"][i])
    new = new.lower()
    new = new.split()

    new = [lemmatizer.lemmatize(word) for word in new if not word in set(stopwords.words('english'))]
    new = ' '.join(new)
    corpus.append(new)
# print(corpus)

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

cv = TfidfVectorizer(max_features = 2500) 
"""
    we are restricting the number of features/words to top 2500 most frequent, because there will be many 
    words that only occur once and do  not place that important of a role
"""

X = cv.fit_transform(data["message"]).toarray()

print(len(X)) 
""" 
    Number of sentences there are in the dataset 
"""
print(len(X[1])) 
"""
    Number of words there are in the dataset 
"""

""" Now we will convert the label column ->'ham' 'spam' to dummy variables i.e 0 and 1's """

y = pd.get_dummies(data["label"])
# print(y) 

"""
we can see that it makes two columns ham and spam where in true and false are mentioned, 
we'll take just one column rather than the whole two columns 
"""
y = y.iloc[:,1].values
# print(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

"""
Getting the accuracy of my model
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy of my model is :",accuracy)


"""
Making an example and testing it out as well
"""
# Preprocess the input sentence
input_sentence = "Hello customer, get 10,000$ discount for free, click on this link www.ikshan.com"
preprocessed_sentence = re.sub('[^a-zA-Z]', ' ', input_sentence)  # Remove non-alphabetic characters
preprocessed_sentence = preprocessed_sentence.lower()  # Convert to lowercase
preprocessed_sentence = preprocessed_sentence.split()  # Tokenize into words
preprocessed_sentence = [lemmatizer.lemmatize(word) for word in preprocessed_sentence if word not in set(stopwords.words('english'))]  # Remove stopwords and lemmatize
preprocessed_sentence = ' '.join(preprocessed_sentence)

# Convert the preprocessed sentence to TF-IDF representation
sentence_tfidf = cv.transform([preprocessed_sentence]).toarray()

# Use the trained model to predict the label
prediction = spam_detect_model.predict(sentence_tfidf)

# Map the prediction to the corresponding label
label = "spam" if prediction[0] == 1 else "ham"

print("The sentence is classified as:", label)
