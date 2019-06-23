from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('customLogout', views.customLogout, name='customLogout'),
    path('stock/<str:toSearchStock>/<str:stockName>', views.stockSearch, name='stockSearch'),
    path('stock/<str:toSearchStock>', views.stockSearch, name='stockSearch'),
    path('forex/<str:toSearchForex>', views.forexSearch, name='forexSearch'),
    path('addPurchase', views.addPurchase, name='addPurchase'),
    path('portfolio', views.portfolio, name='portfolio'),
    path('signUpView', views.signUpView, name='signUpView'),
    path('loginCheckView', views.loginCheckView, name='loginCheckView'),
    path('login', views.loginView, name='loginView'),
    path('signup', views.signUpView, name='signUpView'),
    path('delPurchase/<int:delId>', views.delPurchase, name='delPurchase'),
    path('news', views.newsView, name='newsViews'),
]
