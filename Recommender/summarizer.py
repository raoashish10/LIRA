import pandas as pd
import re
import nltk
import re
data = pd.read_csv(r'../data/Mili_bank_forest_final.csv')
summaries_list = []
for i in range(len(data)):
    article_text = str(list(data['text'])[i])
    article_text = re.sub(r'[[0-9]*]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)

    file1 = open("legal_stopwords.txt","r+")
    legal_stopwords = set(file1.readlines())
    legal_stopwords = [re.sub(r'\n','',i) for i in legal_stopwords]
    lower_legal_stopwords = [re.sub(r'\n','',i.lower()) for i in legal_stopwords]

    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word.lower() in lower_legal_stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 2
                else:
                    word_frequencies[word] += 2
            else:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    import heapq
    summary_sentences = heapq.nlargest(10, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    summaries_list.append(summary)
data['summary'] = summaries_list
data = data[['File Name','summary']]
data.to_csv(r'summaries.csv')