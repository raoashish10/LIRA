from django.shortcuts import render, HttpResponse
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import nltk
import pandas as pd
import os
# Create your views here.

data1 = pd.read_csv(r'./data/Mili_bank_forest_final.csv')
def index(request):
    return render(request, 'index.html')

def results(request):
    return render(request, 'results.html')

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        model = Doc2Vec.load(r"./models/d2v.model")
        test_data = word_tokenize(query.lower())
        v1 = model.infer_vector(test_data)
        similarities = []
        for i in range(len(data1['text'])):
            d2 = model.wv.dv[str(i)]
            similarities.append(np.dot(matutils.unitvec(v1), matutils.unitvec(d2)))
        d2v_df = pd.DataFrame({'filename':list(data1['File Name']),'similarities':similarities})
        results = d2v_df.sort_values(by=['similarities'],ascending = False).head()
        return HttpResponse(results)
    return render(request, 'search.html')

def summary(request):
    return render(request, 'summary.html')