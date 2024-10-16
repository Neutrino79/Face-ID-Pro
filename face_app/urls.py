from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register_face/', views.register_face, name='register_face'),
    path('test_face/', views.test_face, name='test_face'),
]