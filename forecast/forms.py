from django import forms
from django.http import request
from django.utils.safestring import mark_safe
from django.shortcuts import redirect



weight_decay_choice = (
    ('0', '0'),
    ('1e-8', '1e-8'),
)

class formcrnn(forms.Form):
    

    batch_size= forms.IntegerField(label='batch_size', min_value=1, initial=1024 )

    num_layer = forms.IntegerField(label='num_layer', max_value=4, min_value=1, initial=1)

    hidden_size = forms.IntegerField(label='hidden_size', max_value=500, min_value=1, initial=100)

    learning_rate = forms.FloatField(label='learning_rate', max_value=1, min_value=0.1, initial=1)

    weight_decay = forms.ChoiceField(label='weight_decay', choices=weight_decay_choice)
    '''widget=forms.RadioSelect'''
    epoch = forms.IntegerField(label='epoch', max_value=10000, min_value=100, initial=100)

    n1 = forms.IntegerField(label='n1', min_value=1, initial=1)

    n2 = forms.IntegerField(label='n2', min_value=1, initial=2)

    n3 = forms.IntegerField(label='n3', min_value=1, initial=4)


class formlstm(forms.Form):
    

    batch_size= forms.IntegerField(label='batch_size', min_value=1, initial=1024, help_text="請輸入2的次方 e.g. 2、4、8...")

    num_layer = forms.IntegerField(label='num_layer', max_value=4, min_value=1, initial=1, help_text="1~4")

    hidden_size = forms.IntegerField(label='hidden_size', max_value=500, min_value=1, initial=100)

    learning_rate = forms.FloatField(label='learning_rate', max_value=1, min_value=0.1, initial=1)

    weight_decay = forms.ChoiceField(label='weight_decay', choices=weight_decay_choice)
    '''widget=forms.RadioSelect'''
    epoch = forms.FloatField(label='epoch', max_value=10000, min_value=100, initial=100)



IMP_CHOICES = (
    ('1', 'LSTM'),
    ('2', 'CRNN'),
)
#左邊的是VALUE
OPTIONS = (
    ('ETH','ETH'),
    ('BNB','BNB'),
    ('ADA','ADA'),
    ('XRP','XRP'),
    ('DOGE','DOGE'),
    ('LTC','LTC'),
    ('LINK','LINK'),
    ('BCH','BCH'),
    )
class formchoose(forms.Form):
    
    
    crypto_choice = forms.TypedMultipleChoiceField(label='crypto_choice', choices=OPTIONS,widget=forms.CheckboxSelectMultiple())
    
class formmodel(forms.Form):
    model_choice = forms.ChoiceField(label='模型選擇', choices=IMP_CHOICES)
    day = forms.IntegerField(label='想預測的天數', max_value=1460, min_value=1, initial=40)


class formchoice(forms.Form):
    crypto_choice1 = forms.CharField(label='幣種一', max_length=20 )
    crypto_choice2 = forms.CharField(label='幣種二', max_length=20)
    crypto_choice3 = forms.CharField(label='幣種三', max_length=20)