from django.db import models

# Create your models here.
class crnn_parameter(models.Model):
    crypto1 = models.CharField(max_length=50)
    crypto2 = models.CharField(max_length=50)
    crypto3 = models.CharField(max_length=50)
    predict_days = models.IntegerField()
    batch_size= models.IntegerField()

    num_layer = models.IntegerField()

    hidden_size = models.IntegerField()

    learning_rate = models.FloatField()

    weight_decay = models.FloatField()

    epoch = models.FloatField()

    n1 = models.FloatField(null=True)

    n2 = models.FloatField(null=True)

    n3 = models.FloatField(null=True)

    Test_error = models.FloatField()
    

class lstm_parameter(models.Model):
    crypto1 = models.CharField(max_length=50)
    crypto2 = models.CharField(max_length=50)
    crypto3 = models.CharField(max_length=50)
    predict_days = models.IntegerField()
    batch_size= models.IntegerField()

    num_layer = models.IntegerField()

    hidden_size = models.IntegerField()

    learning_rate = models.FloatField()

    weight_decay = models.FloatField()

    epoch = models.FloatField()

    Test_error = models.FloatField()
    
