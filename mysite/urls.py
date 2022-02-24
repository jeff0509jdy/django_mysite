"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from django.conf.urls import include, url   
from forecast.views import dataloader, crnnform, lstm, lstmform, choose , home, test, lstmrecord, crnnrecord, show, choice


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^crnnprediction/$', dataloader),
    url(r'^crnnform/$', crnnform),
    url(r'^lstmprediction/$', lstm),
    url(r'^lstmform/$', lstmform),
    url(r'^choose/$', choose),
    url(r'^choice/$', choice),
    url(r'^$', home),
    url(r'^test/$', test),
    url(r'^show/$', show),
    url(r'^lstmrecord/$', lstmrecord),
    url(r'^crnnrecord/$', crnnrecord),
    
]
