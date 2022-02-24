from datetime import datetime
from django.shortcuts import redirect, render


#def hello_world(request):
#    return render(request, 'hello_world.html', {
#        'current_time': str(datetime.now()),
#    })

from .forms import formcrnn,formlstm,formchoose,formmodel,formchoice
from .models import crnn_parameter, lstm_parameter


def crnnform(request):
    choose2 = request.session.get('choose2')
    day =request.session.get('day')
    request.session['choose2'] = choose2 
    request.session['day'] = day 
    return render(request, 'crnnform.html', {'form':formcrnn})




import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import numpy as np
from torchsummary import summary
import torch.utils.data as Data
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm.notebook import trange





def dataloader(request):
    choose2 = request.session.get('choose2')
    day =int(request.session.get('day'))
    predict_days=day
    df_for_price = crawler(choose2[0],choose2[1],choose2[2])

    col = list(df_for_price.columns)
    #col[0]
    df01 = df_for_price.loc[:, [col[0], col[1], col[2]]]
    df02 = df_for_price.loc[:, [col[1], col[2], col[0]]]
    df03 = df_for_price.loc[:, [col[2], col[0], col[1]]]
    df04 = df_for_price.loc[:, [col[3]]]
    

    form = formcrnn(request.POST)
    
    BATCH_SIZE =int(form['batch_size'].value())
    NUM_LAYER =int(form['num_layer'].value())
    HIDDEN_SIZE =int(form['hidden_size'].value())
    LEARNING_RATE =float(form['learning_rate'].value())
    WEIGHT_DECAY =float(form['weight_decay'].value())
    EPOCH =int(form['epoch'].value())
    n1 =int(form['n1'].value())
    n2 =int(form['n2'].value())
    n3 =int(form['n3'].value())


    
    #df01 = pd.read_excel(r"C:/Users/liu/Desktop/專題/資料/CNN/價格資料_CNN用_5e(剔除400筆後的不含日期之PCA_01)_0630.xlsx")
    #df02 = pd.read_excel(r"C:/Users/liu/Desktop/專題/資料/CNN/價格資料_CNN用_5e(剔除400筆後的不含日期之PCA_02)_0630.xlsx")
    #df03 = pd.read_excel(r"C:/Users/liu/Desktop/專題/資料/CNN/價格資料_CNN用_5e(剔除400筆後的不含日期之PCA_03)_0630.xlsx")
    #df04 = pd.read_excel(r"C:/Users/liu/Desktop/專題/資料/CNN/價格資料_CNN用_5e(剔除400筆後的不含日期之BTC收盤價)_0630.xlsx")
    #df01 = df01[:1064]         
    #df02 = df02[:1064]
    #df03 = df03[:1064]
    #df04 = df04[:1064]
    
    data_x = np.stack((df01,df02,df03),1)    
    total_period = len(data_x)
#    print(total_period)

    data_x = data_x.astype('float')
    data_y = df04.astype('float')
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = ((data_x/data_x[0]) - 1 )
    data_y = ((data_y/data_y[0]) - 1 )   #對y值進行 first value-based normalization
    
    day = len(data_y) - day
    train_size = int(day)                 #1043是20天；1023是40天；988是75天
    test_size = len(data_x) - train_size
#    print(test_size)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:]
    test_y = data_y[train_size:]
    
    
    #############################################################################################
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    #轉換維度
    train_x = train_x.permute(0,2,1)     
    test_x = test_x.permute(0,2,1)

    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()

    data_train = Data.TensorDataset(train_x, train_y)
    data_test = Data.TensorDataset(test_x, test_y)


    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=BATCH_SIZE,shuffle=True)

    
    OPTIMIZER = "Adadelta"        #此行只是為了打印，並不影響模型，所以模型處要重新修改優化器
    

    class LSTM(nn.Module):
        def __init__(self,nIn,nhidden,nOut,num_layer=2):
            super(LSTM,self).__init__()
            self.lstm=nn.LSTM(nIn,nhidden,num_layer)
                         #Sequence batch channels (W,b,c)
            self.out=nn.Linear(nhidden,nOut)
        def forward(self, input):
            output,_=self.lstm(input)
            output=self.out(output[:,:,:])
            return output
    
    
    
    class CRNN(nn.Module):
        def __init__(self,imgC,nclass,nhidden):                                     #論文中卷積 5 層，後面跟著 5 個 ReLU 函數
            super(CRNN,self).__init__()
            cnn = nn.Sequential()
            cnn.add_module('conv{}'.format(0), nn.Conv1d(imgC, n1, 1))              # 參數1：輸入的通道數、參數2：輸出的通道數（即卷積核數量，i.e. 有多少卷積核，就有多少通道）、參數3：捲積核尺寸 
            cnn.add_module('relu{}'.format(0), nn.ReLU(True))
            cnn.add_module('pooling{}'.format(0),nn.MaxPool1d(2))                   #在這裡有 1 層Polling

            cnn.add_module('conv{}'.format(1), nn.Conv1d(n1, n2, 1))                # 輸入的通道數即為上層的卷積核數量
            cnn.add_module('relu{}'.format(1), nn.ReLU(True))
            #cnn.add_module('pooling{}'.format(1),nn.MaxPool1d(2))

            cnn.add_module('conv{}'.format(2), nn.Conv1d(n2, n3, 1))
    #        cnn.add_module('drop{}'.format(2), nn.Dropout(0.5))                    # 一層 Dropout
            cnn.add_module('relu{}'.format(2), nn.ReLU(True))

    #         cnn.add_module('conv{}'.format(3), nn.Conv1d(10, 10, 1))
    #         #cnn.add_module('drop{}'.format(3), nn.Dropout(0.2))
    #         cnn.add_module('relu{}'.format(3), nn.ReLU(True))

    #         cnn.add_module('conv{}'.format(4), nn.Conv1d(10, 10, 1))
    #         #cnn.add_module('drop{}'.format(4), nn.Dropout(0.2))
    #         cnn.add_module('relu{}'.format(4), nn.ReLU(True))

            #cnn.add_module('conv{}'.format(5), nn.Conv1d(320, 640, 1))
            #cnn.add_module('relu{}'.format(5), nn.ReLU(True))
            #cnn.add_module('pooling{}'.format(4),nn.MaxPool1d(2))

            #cnn.add_module('conv{}'.format(6), nn.Conv1d(640, 1280, 1))
            #cnn.add_module('relu{}'.format(6), nn.ReLU(True))

            self.cnn=cnn

            self.rnn=nn.Sequential(
                LSTM(n3,nhidden,nclass,num_layer = NUM_LAYER)                         #第一個參數接CNN輸出的最後一個參數   / num
            )


        def forward(self,input):
            conv = self.cnn(input)
            #print('conv.size():',conv.size())
            b,c,w=conv.size()
            #conv=conv.squeeze(1)#b ,w
            #conv=conv.reshape[-1,1,c]
            conv=conv.permute(0,2,1) #b,w,c
            rnn_out=self.rnn(conv)                    #即為 def__init__()裡面最後一行，該行計算出來的即為經過卷積以及循環後所得出來的最終out
            #print('rnn_out.size():',rnn_out.size())
            #out=F.log_softmax(rnn_out,dim=2)
            #print('out.size():',out.size())
            return rnn_out


    net = CRNN(3, 1, HIDDEN_SIZE)                         # def __init__(self,imgC,nclass,nhidden):  #圖片輸入變數/經過CNN與LSTM後，最終輸出的個數/LSTM每層神經元個數
    if torch.cuda.is_available():
        net.cuda()               #将所有的模型参数移动到GPU上
    optimizer = torch.optim.Adadelta(net.parameters(), lr=LEARNING_RATE, rho=0.9, eps=1e-06, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    #準備訓練
    net.train(mode=True)


    for epoch in trange(EPOCH):
        running_loss = 0.0
        running_acc = 0.0
        #训练
        for i,data in enumerate(data_loader_train,1):
            train_x,train_y = data
            #判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
            if torch.cuda.is_available():
                train_x = Variable(train_x).cuda()
                train_y = Variable(train_y).cuda()
            else:
                train_x = Variable(train_x)
                train_y = Variable(train_y)
            out = net(train_x)
            out = out.squeeze(1)                             # squeeze是用來把維度去掉，在這裡指定把out張量的第二個維度去掉。  https://blog.csdn.net/zenghaitao0128/article/details/78512715
            loss = loss_fn(out,train_y)                      # https://www.jb51.net/article/194848.htm
    #        loss_list.append(loss.item())  #畫損失圖用，沒必要的話就註解掉     

            running_loss += loss.item() * train_y.size(0)
            _, pred = torch.max(out,1)
            num_correct = (pred == train_y.data).sum()
            accuracy = (pred == train_y.data).float().mean()
            running_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
                epoch+1,running_loss/(len(data_train)),running_acc/len(data_train)
            ))

    net.eval()

    #預測值
    test_x,test_y = test_x.float().cuda(),test_y.float().cuda()
    var_x=Variable(test_x)
    var_y=Variable(test_y)
    pred_test = net(var_x)
    pred_test = pred_test.squeeze(1)
    pred_test = pred_test.view(-1).data
    pred__test = pred_test.cpu()

    #df04 = pd.read_excel(r"C:/Users/liu/Desktop/專題/資料/CNN/價格資料_CNN用_5e(剔除400筆後的不含日期之BTC收盤價)_0630.xlsx")
    #df04 = df04[:1064]
    data_y = df04.values

    prediction_value = (pred__test + 1) * data_y[0]

    test_y = (test_y.cpu() + 1) * data_y[0]

    plt.plot(prediction_value.cpu().numpy() , 'r', label='prediction')
    plt.plot(test_y.detach().cpu().numpy() , 'b', label='real')
    plt.legend(loc='best')
    plt.savefig("C:\\Users\\GOD\\coding\\mysite\\static\\forecast_test.jpg")
    plt.close()
    Test_error = round(sqrt(mean_squared_error(prediction_value.cpu(), test_y.cpu())))


    crypto1 = choose2[0]
    crypto2 = choose2[1]
    crypto3 = choose2[2]
    unit = crnn_parameter.objects.create(crypto1=crypto1, crypto2=crypto2, crypto3=crypto3, predict_days=predict_days, batch_size= BATCH_SIZE, num_layer = NUM_LAYER, hidden_size = HIDDEN_SIZE,
    learning_rate = LEARNING_RATE, weight_decay = WEIGHT_DECAY, epoch = EPOCH, n1=n1, n2=n2, n3=n3, Test_error = Test_error)
    unit.save()  #寫入資料庫
    crnnstats = crnn_parameter.objects.all().order_by('Test_error')
    ordernum=count2(crnnstats)
    
    return render(request, 'crnnprediction.html', {
        'Test_error': Test_error, 'ordernum':ordernum, 'all':len(crnnstats),
    })

    
    


def lstmform(request):
    choose2 = request.session.get('choose2')
    day =request.session.get('day')
    request.session['choose2'] = choose2 
    request.session['day'] = day 
    return render(request, 'lstmform.html', {'form':formlstm})



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torchsummary import summary
import torch.utils.data as Data
import os

def lstm(request):
    
    choose2 = request.session.get('choose2')
    day =int(request.session.get('day'))
    predict_days=day
    df_for_price = crawler(choose2[0],choose2[1],choose2[2])


    form = formlstm(request.POST)
    BATCH_SIZE =int(form['batch_size'].value())
    NUM_LAYER =int(form['num_layer'].value())
    HIDDEN_SIZE =int(form['hidden_size'].value())
    LEARNING_RATE =float(form['learning_rate'].value())
    WEIGHT_DECAY =float(form['weight_decay'].value())
    EPOCH =int(form['epoch'].value())
    
    col = list(df_for_price.columns)
    data_x_origin = df_for_price.loc[:, [col[0],col[1],col[2]]]
    data_y_origin = df_for_price.loc[:, [col[3]]]
    data_x = data_x_origin.values
    data_y = data_y_origin.values
    
    data_x = ((data_x/data_x[0]) - 1 )
    data_x = np.array(data_x)
    #data_x = np.array(data_x_normed)
    data_y = ((data_y/data_y[0]) - 1 )   #對y值進行 first value-based normalization
    data_y = np.array(data_y)
    #print(data_y)
    #plt.plot(data_y)
    total_period = len(data_x)

    
    day = len(data_y) - day
    train_size = int(day)                #訓練樣本數目
    test_size = len(data_x) - train_size  #測試樣本數目
    #print(test_size)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    test_x = data_x[train_size:]
    test_y = data_y[train_size:]

    # 偷看一下運行結果... 
    # print(train_x)
    # print("------------")
    # train_x = train_x.reshape(-1, 1)
    # print(train_x)
    # print("------------")

    train_x = train_x.reshape(-1, 1, 3)    #總共有三個維度：我不知道第一個維度有多少(i.e. 我不知道有多少日的價格資料，我要讓程式自己去跑這個數字)，但我知道在第一個維度下有1串資料，每一串資料有10個特徵

    '''
    分析程式背後含義 (若不懂則建議一行一行執行)
    print(train_x)
    print(train_x[0])
    print(train_x[0][0])
    print(len(train_x))                    # 共有3000天的資料
    print(len(train_x[0]))                 # 每天共有一串資料
    print(len(train_x[0][0]))              # 每串資料共有十個特徵
    '''

    train_y = train_y.reshape(-1, 1)        #把它變成3600列、然後只有1行的概念
    train_x = torch.from_numpy(train_x)     #把array轉成tensor格式 https://pytorch.org/docs/stable/generated/torch.from_numpy.html
    train_y = torch.from_numpy(train_y)

    test_x = test_x.reshape(-1, 1, 3)
    test_y = test_y.reshape(-1, 1)
    test_x = torch.from_numpy(test_x)     
    test_y = torch.from_numpy(test_y)

    data_train = Data.TensorDataset(train_x, train_y)
    data_test = Data.TensorDataset(test_x, test_y)

    
    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=BATCH_SIZE,shuffle=True)

    
    class NET(nn.Module):
        def __init__(self,input_size=3,hidden_size = HIDDEN_SIZE,output_size = 1,num_layer = NUM_LAYER): # hidden_size：隱藏層神經元個數是我們自己設定的
            super(NET,self).__init__()                                              #允許使用者在子類中調用超類的方法
            self.rnn=nn.LSTM(input_size,hidden_size,num_layer)                      #定義 LSTM
            self.out=nn.Linear(hidden_size,output_size)                             #设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量   https://blog.csdn.net/qq_42079689/article/details/102873766
        def forward(self,x):
            out,_=self.rnn(x)
            out=self.out(out[:,-1,:])
            return out

    net = NET()      # 啟動 Net
    
    OPTIMIZER = "Adadelta"
    

    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=LEARNING_RATE, rho=0.9, eps=1e-06, weight_decay=WEIGHT_DECAY)  # 補1：先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
    loss_func = torch.nn.MSELoss()        #均方損失函數                                                       # 補2：優化器 https://blog.csdn.net/Ibelievesunshine/article/details/99624645
    if torch.cuda.is_available():
        net.cuda()                        # 将所有的模型参数移动到 GPU 上， i.e. 将数据的形式变成 GPU 能读的形式，然后将 CNN 也变成 GPU 能读的形式之概念

    net.train(mode=True)


    

    loss_list = []
    for epoch in range(EPOCH):              # 對網絡訓練 7000 次
        train_x,train_y = train_x.float().cuda(),train_y.float().cuda()    # 将数据(和网络)都推到 GPU，接上.cuda()
        var_x=Variable(train_x)            # 用Variable包一下，才可在GPU上做運算
        var_y=Variable(train_y)
        out = net(var_x) 
        loss = loss_func(out, var_y)       # 計算損失函數
        loss_list.append(loss.item())
        
        optimizer.zero_grad()              # 意思是把梯度置零，也就是把loss关于weight的导数变成0. https://blog.csdn.net/scut_salmon/article/details/82414730
        loss.backward()                    # i.e. 反向传播求梯度
        optimizer.step()                   # 更新所有參數
        if (epoch + 1) % 100 == 0:         
            print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.data))

            
            

    
    net.eval()

    test_x,test_y = test_x.float().cuda(),test_y.float().cuda()
    var_x=Variable(test_x)
    var_y=Variable(test_y)
    pred_test = net(var_x)
    pred_test = pred_test.view(-1).data                          #view()函数作用是将一个多行的Tensor,拼接成一行。 https://blog.csdn.net/program_developer/article/details/82112372
    #pred_test=torch.max(pred_test,1)[1].data.numpy().squeeze()




    pred__test = pred_test.cpu()

    data_y = data_y_origin.values

    #數據轉換
    prediction_value = (pred__test + 1) * data_y[0]
    test_y = (test_y.cpu() + 1) * data_y[0]
    

    plt.plot(prediction_value.cpu().numpy() , 'r', label='prediction')
    plt.plot(test_y.detach().cpu().numpy() , 'b', label='real')
    plt.legend(loc='best')
    plt.savefig("C:\\Users\\GOD\\coding\\mysite\\static\\forecast_test_lstm.jpg") 
    plt.close()

    #calculate RMSE
    Test_error = round(sqrt(mean_squared_error(prediction_value.cpu(), test_y.cpu())))

    
    crypto1 = choose2[0]
    crypto2 = choose2[1]
    crypto3 = choose2[2]
    unit = lstm_parameter.objects.create(crypto1=crypto1, crypto2=crypto2, crypto3=crypto3, predict_days=predict_days, batch_size= BATCH_SIZE, num_layer = NUM_LAYER, hidden_size = HIDDEN_SIZE,
    learning_rate = LEARNING_RATE, weight_decay = WEIGHT_DECAY, epoch = EPOCH, Test_error = Test_error) 
    unit.save()  #寫入資料庫
    lstmstats = lstm_parameter.objects.all().order_by('Test_error')

    ordernum=count1(lstmstats)
    
    return render(request, 'lstmprediction.html', {
        'Test_error': Test_error, 'ordernum':ordernum, 'all':len(lstmstats), 
    })



from django.core.exceptions import ValidationError

def choose(request):
    form = formchoose(request.POST or None)
    if request.method == 'POST':
            
        if form.is_valid(): 
            choose2= form.cleaned_data["crypto_choice"]
            request.session['choose2'] = choose2  
            
            if  len(choose2) != 3:
                raise ValidationError(('請選擇3個幣種'))
            else:
                return redirect('/show/')
            
            
    return render(request, 'choose.html', {'form':formchoose})


    
def show(request):
    choose2 = request.session.get('choose2')
    choose2 = [x.upper() for x in choose2]
    df_for_price = crawler(choose2[0],choose2[1],choose2[2])
    if len(df_for_price)==0:
        return render(request, 'show.html', {'資料天數': len(df_for_price)})
    else:
        datashow(df_for_price,choose2[0],choose2[1],choose2[2])
        
    form = formmodel(request.POST or None)
    if request.method == 'POST':    
        if form.is_valid(): 
            model_choice=form.cleaned_data["model_choice"]
            day=form.cleaned_data["day"]
            request.session['choose2'] = choose2
            request.session['day'] = day
            if model_choice=='1':
                return redirect('/lstmform/')

            elif model_choice=='2':
                return redirect('/crnnform/')
    
    return render(request, 'show.html', {'form':formmodel,'資料天數': len(df_for_price)}) 
        




def test(request):
    choose1 = request.session.get('choose1')
    choose2 = request.session.get('choose2')
    #choose =form['IMP_CHOICES'].value()
    
    return render(request, 'test.html', {'choose1': choose1, 'choose2': choose2}) 



def home(request):
    return render(request, 'home.html')



import requests
from datetime import datetime
import yfinance as yf

def crawler(crypto_1, crypto_2, crypto_3, crypto_4="BTC"): 
    crypto1 = yf.Ticker(str(crypto_1) + "-USD")
    crypto1_data = crypto1.history(start="2018-01-02", end="2022-01-01")

    crypto2 = yf.Ticker(str(crypto_2) + "-USD")
    crypto2_data = crypto2.history(start="2018-01-02", end="2022-01-01")

    crypto3 = yf.Ticker(str(crypto_3) + "-USD")
    crypto3_data = crypto3.history(start="2018-01-02", end="2022-01-01")

    crypto4 = yf.Ticker(str(crypto_4) + "-USD")
    crypto4_data = crypto4.history(start="2018-01-02", end="2022-01-01")

    
    df_for_price = pd.DataFrame({str(crypto_1)+"_Close": crypto1_data["Close"],
                                str(crypto_2)+"_Close": crypto2_data["Close"],
                                str(crypto_3)+"_Close": crypto3_data["Close"],
                                str(crypto_4)+"_Close": crypto4_data["Close"]
                                })
    df_for_price.dropna(axis = 0, inplace = True)


    #print(df_for_price)
    return df_for_price


def datashow(df_for_price,crypto_1, crypto_2, crypto_3):
    col = list(df_for_price.columns)
    data_x1= df_for_price.loc[:, [col[0]]]
    data_x2= df_for_price.loc[:, [col[1]]]
    data_x3= df_for_price.loc[:, [col[2]]]
    data_y = df_for_price.loc[:, [col[3]]]
    
    plt.plot(data_x1 , '#4169E1', label=crypto_1, linewidth=1)
    plt.plot(data_x2 , '#3CB371', label=crypto_2, linewidth=1)
    plt.plot(data_x3 , '#CD5C5C', label=crypto_3, linewidth=1)
    plt.plot(data_y  ,  label='BTC' ,linewidth=1)
    plt.legend()
    plt.savefig("C:\\Users\\GOD\\coding\\mysite\\static\\資料期間.jpg") 
    plt.close()
   



    
def lstmrecord(request):
    lstmstats = lstm_parameter.objects.all().order_by('Test_error')
    

    return render(request, 'lstmrecord.html', locals())


def crnnrecord(request):
    
    crnnstats = crnn_parameter.objects.all().order_by('Test_error')

    return render(request, 'crnnrecord.html', locals())


def choice(request):
    form = formchoice(request.POST or None)
    if request.method == 'POST':
            
        if form.is_valid(): 
            choice1= form.cleaned_data["crypto_choice1"]
            choice2= form.cleaned_data["crypto_choice2"]
            choice3= form.cleaned_data["crypto_choice3"]
            choose2=[choice1,choice2,choice3]
            request.session['choose2'] = choose2
            
            if  len(choose2) != 3:
                raise ValidationError(('請選擇3個幣種'))
            else:
                return redirect('/show/')
            
            
    return render(request, 'choice.html', {'form':formchoice})


def count1(lstmstats):
    K=len(lstmstats)
    b=1
    for i in range(0,len(lstmstats)):
        if lstmstats[i]==lstm_parameter.objects.get(id=K):
            break
        else:
            b+=1
    return b

def count2(crnnstats):
    K=len(crnnstats)
    b=1
    for i in range(0,len(crnnstats)):
        if crnnstats[i]==crnn_parameter.objects.get(id=K):
            break
        else:
            b+=1
    return b