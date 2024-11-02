from django.urls import path
from . import views

urlpatterns = [
    path('fourier/', views.fourier_view, name='fourier_view'),
]