from django.shortcuts import render
from django.http import HttpResponse
import joblib
import pandas as pd
import os
from django.views.decorators.csrf import csrf_exempt

def hello(request):
    return HttpResponse("Hello, world!")

# get the directory that this script is in
dir_path = os.path.dirname(os.path.realpath(__file__))

# combine that directory with the filename to create the full path
model_path = os.path.join(dir_path, 'kmeans_model.pkl')

# Load the model
model = joblib.load(model_path)

@csrf_exempt
def predict(request):
    # Ensure we're working with a POST request
    if request.method == 'POST':
        data = request.POST.dict()  # Convert QueryDict to regular dict
        data.pop('csrfmiddlewaretoken', None)  # Remove CSRF token

        # reshape the data for the model
        data_df = pd.DataFrame(data, index=[0])

        # Use your model to make a prediction
        prediction = model.predict(data_df)
        
        # Convert prediction to HttpResponse or JsonResponse
        return HttpResponse(f'The predicted species is: {prediction[0]}')
        
    else:
        return HttpResponse('This endpoint expects a POST request.')

def input_form_view(request):
    return render(request, 'input_form.html')