from django.urls import path
from . import views

app_name = 'ml_app'

urlpatterns = [
    path('hello/', views.hello, name='hello'),
    path('form/', views.input_form_view, name='input_form'),
    path('predict/', views.predict, name='predict'),
]
