from django.contrib import admin

# Register your models here.
from .models import crnn_parameter, lstm_parameter

admin.site.register(crnn_parameter)
admin.site.register(lstm_parameter)
