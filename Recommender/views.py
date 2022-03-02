from django.shortcuts import render, HttpResponse, redirect
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import nltk
import pandas as pd
import os
import numpy as np
from gensim import matutils 
# Create your views here.

data1 = pd.read_csv(r'./data/Mili_bank_forest_final.csv')
def index(request):
    return render(request, 'index.html')

def results(request):
    result = request.session['result']
    file_dict = result['filename']
    similarities_dict = result['similarities']
    file_similarities = []
    for i in file_dict:
        file_similarities.append([file_dict[i],round(similarities_dict[i],4)])
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
        return redirect('results')
        # return HttpResponse(results)
    return render(request, 'search.html')

def summary(request):
    casename = request.GET.get('filename')
    case_details = pd.read_csv(r'./data/case_details.csv')
    case_row = case_details[case_details['File Name'] == casename].head(1)
    data_case_name =  case_row['Case Name'].values[0]
    involved = case_row['Involved Personell'].values[0]
    date_decided = case_row['Date (Decided)'].values[0]
    court = case_row['Court'].values[0]
    category = case_row['category'].values[0]
    if category == 62:
        category = 'Military'
    elif category == 6:
        category = 'Banking'
    else:
        category = 'Environment'
    return render(request, 'summary.html',{'data_case_name':data_case_name,'involved':involved,'date_decided':date_decided,'court':court,'category':category})