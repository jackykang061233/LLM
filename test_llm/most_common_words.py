import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
 
# Download required NLTK resources
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
 
# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
 
# Function to convert NLTK's part-of-speech tags to WordNet's part-of-speech tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown
 
def get_top_10_common_words(text, additional_stopwords=None):
    # Set of English stop words
    stop_words = set(stopwords.words('english'))

    if additional_stopwords:
        stop_words.update(additional_stopwords)
   
    # Tokenize the text
    tokens = word_tokenize(text)
   
    # Filter out stop words and punctuation
    words = [word for word in tokens if word.isalpha() and word.lower() not in stop_words]
   
    # Get part-of-speech tags
    pos_tags = pos_tag(words)
   
    # Lemmatize each word with its part-of-speech tag
    lemmatized_words = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos)) 
                        for word, pos in pos_tags
                        if get_wordnet_pos(pos) != wordnet.VERB
                       ]
   
    # Count the occurrences of each lemmatized word
    word_counts = Counter(lemmatized_words)
   
    # Get the most common 10 words
    top_10 = word_counts.most_common(10)
   
    return top_10
 
with open('doc/fed.txt') as f:
    text = f.read()

additional_stopwords = {'year', 'uh', 'um'}
# Get the top-10 most common words
top_10_words = get_top_10_common_words(text)
 
# Output the result
print(top_10_words)