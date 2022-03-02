from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'index.html')

def results(request):
    return render(request, 'results.html')

def search(request):
    return render(request, 'search.html')

def summary(request):
    return render(request, 'summary.html')