from django.shortcuts import render, HttpResponse, redirect
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import re
import math
import time
import nlpcloud
import numpy as np
from gensim import matutils 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import nlpcloud

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
# Create your views here.
# model_name = 'google/pegasus-large'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

client = nlpcloud.Client("bart-large-cnn", "f9c60f3620a2bdba081394bd77f44445c2a019b0")

data1 = pd.read_excel(r'./data/Mili_bank_forest_final.xlsx')
summaries_data = pd.read_csv(r'./data/summaries.csv')
summaries_data['summary'].fillna('', inplace= True)
raw_data = pd.read_csv(r'./data/raw.csv',error_bad_lines=False, engine="python")

def index(request):
    return render(request, 'index.html')

def get_similar_docs(cluster,casename):
    if cluster ==  62:
        cluster = 2
    elif cluster == 6:
        cluster = 1
    elif cluster == 30:
        cluster = 0
    chosen = casename
    model = Doc2Vec.load(r"./models/d2v_old.model")
    chosen_row = data1[data1['File Name'] == casename].head(1)
    chosen_text = chosen_row['text'].values[0]
    test_data = word_tokenize(str(chosen_text).lower())
    v1 = model.infer_vector(test_data)
    
    similarities = []
    files = []
    for i in range(len(data1['text'])):
        if data1['Predicted_category'].iloc[i] == cluster and data1['File Name'].iloc[i] != casename:
            d2 = model.dv[str(i)]
            similarities.append(np.dot(matutils.unitvec(v1), matutils.unitvec(d2)))
            files.append(data1['File Name'].iloc[i].split(r".")[0])
    d2v_df = pd.DataFrame({'filename':files,'similarities':similarities})
    results = d2v_df.sort_values(by=['similarities'],ascending = False).head()
    result = results.to_dict()
    file_dict = result['filename']
    similarities_dict = result['similarities']
    file_similarities = []
    for i in file_dict:
        file_similarities.append([file_dict[i],round(similarities_dict[i],4)])
    return file_similarities

def results(request):
    result = request.session['result']
    file_dict = result['filename']
    similarities_dict = result['similarities']
    file_similarities = []
    for i in file_dict:
        file_similarities.append([file_dict[i],round(similarities_dict[i],4)])
    if 'query' in request.COOKIES:
        response = render(request, 'results.html',{'file_similarities':file_similarities,'query':request.COOKIES['query'] })
        response.set_cookie(key='query', value=request.COOKIES['query'])
        return response
    return render(request, 'results.html',{'file_similarities':file_similarities})

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        model = Doc2Vec.load(r"./models/d2v_old.model")
        test_data = word_tokenize(query.lower())
        v1 = model.infer_vector(test_data)
        similarities = []
        for i in range(len(data1['text'])):
            d2 = model.dv[str(i)]
            similarities.append( round( 100 * np.dot(matutils.unitvec(v1), matutils.unitvec(d2)),2) )
        d2v_df = pd.DataFrame({'filename':list(data1['File Name']),'similarities':similarities})
        results = d2v_df.sort_values(by=['similarities'],ascending = False).head()
        request.session['result'] = results.to_dict()
        response = redirect('results') # django.http.HttpResponse
        response.set_cookie(key='query', value=query)
        return response
        # return redirect('results')
        # return HttpResponse(results)
    return render(request, 'search.html')

def abstract_summary(casename):
    print(casename in set(summaries_data['File Name']))
    if casename in set(summaries_data['File Name']):
        case_row = summaries_data[summaries_data['File Name'] == casename].head(1)
        if not case_row['summary'].values[0] == '':
            abstract = case_row['summary'].values[0]
            return abstract
    
    chosen_row = raw_data[raw_data['File Name'] == casename].head(1)
    if chosen_row['text'].values:
        article_text = chosen_row['text'].values[0]
        texts = sent_tokenize(article_text)
    else:
        chosen_row = data1[data1['File Name'] == casename].head(1)
        article_text = chosen_row['text'].values[0]
        texts = split_into_sentences(article_text)
        
    i = 0
    steps = 40
    summary_text = ''
    print('here: ',texts)
    while i < len(texts):
        print(f"{round(100 * (i / len(texts)),2)}%")
        if i+steps < len(texts):
            res = client.summarization(''.join(texts[i:i + steps]))
            summary_text += res['summary_text'] + ' '
        else:
            res = client.summarization(''.join(texts[i:]))
            summary_text += res['summary_text'] + ' '
        print("Sleep time")
        time.sleep(20)
        i += steps
    summary_row = summaries_data[summaries_data['File Name'] == casename].head(1).index
    summaries_data.at[summary_row,'summary'] = summary_text
    summaries_data.to_csv(r'./data/summaries.csv')
    # article_text = re.sub(r'[[0-9]*]', ' ', article_text)
    # article_text = re.sub(r'\s+', ' ', article_text)
    

    # formatted_article_text = re.sub(r'\s+', ' ', article_text)
    # texts = sent_tokenize(formatted_article_text)
    # tgt_text_list = []
    # i = 0
    # steps = 2
    # # res = ''
    # summary_text = ''
    # while i < len(texts):
    #     if i+steps < len(texts):
    #        res = client.summarization(''.join(texts[i:i + steps]))
    #        summary_text += res['summary_text'] + ' '
    #     else:
    #         res = client.summarization(''.join(texts[i:]))
    #         summary_text += res['summary_text'] + ' '
        
    #     i += steps

    # #         print(i)
    #         for j in texts[i:i+steps]:
    #             print('\n\n',j)
    #         batch = tokenizer.prepare_seq2seq_batch(''.join(texts[i:i+steps]), truncation=True, padding='longest', return_tensors="pt").to(torch_device)
    #         translated = model.generate(**batch)
    #         tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    #         if tgt_text not in tgt_text_list:
    #             tgt_text_list.extend(tgt_text)
    #     else:
    #         batch = tokenizer.prepare_seq2seq_batch(''.join(texts[i:]), truncation=True, padding='longest', return_tensors="pt").to(torch_device)
    #         translated = model.generate(**batch)
    #         tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    #         if tgt_text not in tgt_text_list:
    #             tgt_text_list.extend(tgt_text)
    #     i = i + steps
    return summary_text

def summary(request):
    if 'query' in request.COOKIES:
        query = request.COOKIES['query']
    casename = request.GET.get('filename')
    case_details = pd.read_csv(r'./data/case_details.csv')
    # summary_file = pd.read_csv(r'./data/summaries.csv')
    # summary_row = summary_file[summary_file['File Name'] == casename].head(1)
    # summary = summary_row['summary'].values[0]
    case_row = case_details[case_details['File Name'] == casename].head(1)
    
    data_case_name =  case_row['Case Name'].values[0]
    involved = case_row['Involved Personell'].values[0]
    date_decided = case_row['Date (Decided)'].values[0]
    court = case_row['Court'].values[0]
    category = case_row['category'].values[0]
    similar_docs = get_similar_docs(category,casename)
    if category == 62:
        category = 'Military'
    elif category == 6:
        category = 'Banking'
    else:
        category = 'Environment'
    abstract = abstract_summary(casename)
    if math.isnan(court): court = ''
    if  math.isnan(date_decided): date_decided = ''
    if  math.isnan(involved): involved = ''
    if  math.isnan(data_case_name): data_case_name = ''
    return render(request, 'summary.html',{'abstract_summary':abstract,'similar_docs':similar_docs,'data_case_name':data_case_name,'involved':involved,'date_decided':date_decided,'court':court,'category':category,'query':query})
