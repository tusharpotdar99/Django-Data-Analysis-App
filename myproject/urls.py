"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.upload_csv, name='upload_csv'),
    path('data/', views.data, name='data'),
    path('handle_missing_values/', views.handle_missing_values, name='handle_missing_values'),
    path('statistics/original/', views.statistics_original, name='statistics_original'),
    path('statistics/cleaned/', views.statistics_cleaned, name='statistics_cleaned'),
    path('plot-page/', views.show_plot_page, name='show_plot_page'),
    path('plot/', views.generate_plot, name='generate_plot'),
    path('kdeplot/', views.generate_kdeplot, name='generate_kdeplot'),
    path('scatterplot/', views.generate_scatterplot, name='generate_scatterplot'),
    path('barplot/', views.generate_barplot, name='generate_barplot'),
    path('piechart/', views.generate_piechart, name='generate_piechart'),
    path('boxplot/', views.generate_boxplot, name='generate_boxplot'),
    path('heatmap/', views.generate_heatmap, name='generate_heatmap'),
    path('lineplot/', views.generate_lineplot, name='generate_lineplot'),
    path('about/', views.about, name='about'),
    path('report/', views.report, name='report'),
    path('get_stat/', views.get_stat, name='get_stat'),
    path('clean_stat/', views.clean_stat, name='clean_stat'),
    path('show_plot/', views.show_plot, name='show_plot'),
    path('custom_plot/', views.custom_plot, name='custom_plot'),


]
