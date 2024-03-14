import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import re
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
df = pd.read_csv('C:/Users/DELL/Documents/Tweets.csv') 
df.head()
from nltk.corpus import stopwords
#preprocessing the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#assuming 'stop_words' and 'Lemmatizer' are already defined
def preprocess_text(text):
    #convert to lowercase
    text = text.lower()
    
    #remove numbers and special charaters
    text = re.sub(r'\d+', ' ',text)
    text = re.sub(r'[^a-zA-Z\s]', ' ',text)
    
    #tokenize the text
    tokens = nltk.word_tokenize(text)
   
    #remove stoke words
    tokens= [token for token in tokens if token not in stop_words]
    
    #lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    #join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
    nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
df.text
df["clean_text"] = df["text"].apply(lambda s: ' '.join(re.sub("(\w+:\/\/\S+)", " ", str(s)).split()))

# Check the length of your DataFrame
print(len(df))

# Display the text and clean_text columns for a specific row if the index is within the range
index_to_display = 94807
if index_to_display < len(df):
    print(df[['text', 'clean_text']].iloc[index_to_display])
else:
    print("Index out of range.")
df["clean_text"] = df["clean_text"].apply(lambda s: ' '.join(re.sub("[.,!?:;-=""...@#_]"," ",str(s)).split()))

# Display the text and clean_text columns for a specific row
df[['text','clean_text']]
df["clean_text"].replace('\d+',' ',regex=True, inplace=True)
df[['text','clean_text']]
def deEmojify(inputString):
    return inputString.encode('ascii','ignore').decode('ascii')
df["clean_text"] = df["clean_text"].apply(lambda s: deEmojify(s))
df[['text','clean_text']].iloc[12]

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))
stop
import nltk
from nltk.tokenize import word_tokenize

# Assuming you have a piece of text stored in a variable called text
text = "This is a sample sentence."

# Tokenize the text into words
word_tokens = word_tokenize(text)

# Print the tokenized words
print(word_tokens)
def rem_en(input_txt, word_tokens, stop_words):
    words = input_txt.lower().split()
    noise_free_words = [w for w in word_tokens if not w in stop_words]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text
cleaned_text = rem_en("text" ,word_tokens, stop_words)
# Assuming word_tokens and stop_words are defined elsewhere
word_tokens = [...]  # Define or obtain word_tokens
stop_words = [...]   # Define or obtain stop_words

# Apply the rem_en function to the "clean_text" column with word_tokens and stop_words as additional arguments
df["clean_text"] = df["clean_text"].apply(lambda s: rem_en(s, word_tokens, stop_words))

# Display the "text" and "clean_text" columns
print(df[['text', 'clean_text']])
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'\w+')
df["clean_text"] = df["clean_text"].apply(lambda x: token.tokenize(x)) 
df[['text','clean_text']]
# Preprocess the text and store it in a new column
df['preprocessed_text'] = df['airline_sentiment'].apply(preprocess_text)

# Access both 'artist.name' and 'preprocessed_text' columns
df[['airline_sentiment', 'preprocessed_text']]
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
print(f"sentiment scores: {sentiment_scores}")
sentiment_labels = []
for preprocessed_text in df['preprocessed_text']:
    sentiment_scores = sia.polarity_scores(preprocessed_text)
    if sentiment_scores['compound'] > 0.05:
        sentiment_labels.append('Positive')
    elif sentiment_scores['compound'] < -0.05:
        sentiment_labels.append('Negative')
    else:
        sentiment_labels.append('Neutral')
print(f"sentiment scores: {sentiment_label}")
def get_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    sentiment = 1 if sentiment_scores ['pos'] > 0 else 0
    return sentiment
df['sentiment'] = df['preprocessed_text'].apply(get_sentiment)
df
pip install styleCloud
import stylecloud
text = ' '.join(df['text'])
wordcloud = WordCloud(width=267, height=100, max_words=100, background_color='white').generate(text)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()
from PIL import Image
texts=' '.join(df['text'])
starlogo=stylecloud.gen_stylecloud(text=texts,
                                       icon_name='fas fa-star',size=756,
                                       palette='cartocolors.qualitative.Pastel_3',
                                       background_color='white',output_name='starstylecloud.png')
image=Image.open('starstylecloud.png')
image.show()
treelogo=stylecloud.gen_stylecloud(text=texts,
                                       icon_name='fas fa-tree',size=756,
                                       palette='cartocolors.qualitative.Pastel_3',
                                       background_color='white',output_name='treestylecloud.png')
image=Image.open('treestylecloud.png')
image.show()
