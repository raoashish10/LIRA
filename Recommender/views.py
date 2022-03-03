from django.shortcuts import render, HttpResponse, redirect
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import re
import os
import numpy as np
from gensim import matutils 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# Create your views here.
model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

data1 = pd.read_csv(r'./data/Mili_bank_forest_final.csv')
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
    test_data = word_tokenize(chosen_text.lower())
    v1 = model.infer_vector(test_data)
    
    similarities = []
    files = []
    for i in range(len(data1['text'])):
        if data1['Predicted_category'].iloc[i] == cluster and data1['File Name'].iloc[i] != casename:
            d2 = model.dv[str(i)]
            similarities.append(np.dot(matutils.unitvec(v1), matutils.unitvec(d2)))
            files.append(data1['File Name'].iloc[i])
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
        response = render(request, 'results.html',{'file_similarities':file_similarities })
        response.set_cookie(key='query', value=request.COOKIES['query'])
        return response
    return render(request, 'results.html',{'file_similarities':file_similarities })

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        model = Doc2Vec.load(r"./models/d2v_old.model")
        test_data = word_tokenize(query.lower())
        v1 = model.infer_vector(test_data)
        similarities = []
        for i in range(len(data1['text'])):
            d2 = model.dv[str(i)]
            similarities.append(np.dot(matutils.unitvec(v1), matutils.unitvec(d2)))
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
    chosen_row = data1[data1['File Name'] == casename].head(1)
    article_text = chosen_row['text'].values[0]
    article_text = re.sub(r'[[0-9]*]', ' ', article_text)

    article_text = re.sub(r'\s+', ' ', article_text)
    # Removing special characters and digits
    # formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    # print(formatted_article_text)
    formatted_article_text = re.sub(r'\s+', ' ', article_text)
    texts = sent_tokenize(formatted_article_text)
    tgt_text_list = []
    i = 0
    while i < len(texts):
        if i+10 < len(texts):
            batch = tokenizer.prepare_seq2seq_batch(''.join(texts[i:i+10]), truncation=True, padding='longest', return_tensors="pt").to(torch_device)
            translated = model.generate(**batch)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            tgt_text_list.extend(tgt_text)
        else:
            batch = tokenizer.prepare_seq2seq_batch(''.join(texts[i:]), truncation=True, padding='longest', return_tensors="pt").to(torch_device)
            translated = model.generate(**batch)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            tgt_text_list.extend(tgt_text)
        i = i + 10
    return ''.join(tgt_text_list)

def summary(request):
    if 'query' in request.COOKIES:
        query = request.COOKIES['query']
    casename = request.GET.get('filename')
    case_details = pd.read_csv(r'./data/case_details.csv')
    summary_file = pd.read_csv(r'./data/summaries.csv')
    summary_row = summary_file[summary_file['File Name'] == casename].head(1)
    summary = summary_row['summary'].values[0]
    case_row = case_details[case_details['File Name'] == casename].head(1)
    
    data_case_name =  case_row['Case Name'].values[0]
    print(case_row)
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
    return render(request, 'summary.html',{'abstract_summary':abstract,'similar_docs':similar_docs,'data_case_name':data_case_name,'involved':involved,'date_decided':date_decided,'court':court,'category':category,'summary':summary})