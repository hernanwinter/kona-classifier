from django.urls import path

from . import views

urlpatterns = [
    path('poster', views.classify_poster, name='classify'),
]
