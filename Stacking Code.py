# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import models, optimizers, layers, regularizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
num = []
k = 0

def build_model_final(parameters):
    model =  models.Sequential()
    model.add(layers.Dense(1, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(3, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(2, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(1, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(4, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(1))
    adam = optimizers.Adam(lr=0.04, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.002)
    model.compile(optimizer=adam, loss='mse' , metrics=['mae'])
    return model



def build_model(parameters):
    model =  models.Sequential()
    model.add(layers.Dense(int(parameters*2.0), activation='elu', kernel_regularizer=regularizers.l1_l2(0.0001),input_shape=(parameters,)))
    model.add(BatchNormalization())
    model.add(layers.Dense(parameters*3, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(parameters*2, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(parameters, activation='elu'))
    model.add(BatchNormalization())

    model.add(layers.Dense(4, activation='elu'))
    model.add(BatchNormalization())
    model.add(layers.Dense(1))
    adam = optimizers.Adam(lr=0.04, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.002)
    model.compile(optimizer=adam, loss='mse' , metrics=['mae'])
    return model

k = 5
flag = 0
header1 = "Predicted"
header2 = "Measured"
if (flag == 0):
	header1 = header1 + " NOx"
	header2 = header2 + " NOx"
elif (flag == 1):
	header1 = header1 + " FSN"
	header2 = header2 + " FSN"
elif (flag == 2):
	header1 = header1 + " bsfc"
	header2 = header2 + " bsfc"

num_epochs = 20000 # 학습 반복횟수

all_data = pd.read_csv('newData.csv', skiprows=2)
mat = all_data.values

(tot_height, tot_width)= mat.shape

testlen=int(tot_height*0.2) # 20%
trainlen=tot_height-testlen # 80%

all_data_random = shuffle(mat) # data 섞는것

test_data = all_data_random[:testlen, :] # 0~71
train_data = all_data_random[testlen:, :]

train_targets_data = train_data[:,0:1]
train_targets_data_1 = train_data[:, 0:1]
test_targets_data = test_data[:, 0:1]

train_data = train_data[:,3:28] #71~끝 에서 EGR 값을 빼고 저장
test_data = test_data[:,3:28] # 0~71 에서 EGR 값을 빼고 저장

early_stopping_1 = EarlyStopping(monitor='val_loss',min_delta = 5, patience=300,mode = 'auto')
early_stopping = EarlyStopping(monitor='val_loss',min_delta = 5, patience=300,mode = 'auto')
best_model_2 =  ModelCheckpoint(filepath= 'model_check_6.h5', monitor = 'val_loss',save_best_only=True)

print("-------------------------------------------------")
print("train data=",train_data.shape,train_targets_data.shape)
print("test_data=",test_data.shape)
num_val_samples = len(train_data) // k 

#Data Standardization
mean = train_data.mean(axis=0)
train_data -= mean
std =  train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
train_data_0 = train_data[0 : 111 , : ] #5번으로 나눔  
train_data_1 = train_data[111 : 222 , : ]
train_data_2 = train_data[222 : 333 , : ]
train_data_3 = train_data[333 : 444 , : ]
train_data_4 = train_data[444 : 555 , : ]

train_targets_data_0 =  train_targets_data[ 0 :111 , 0:1 ] 
train_targets_data_1 =  train_targets_data[ 111 : 222 , 0:1 ] 
train_targets_data_2 =  train_targets_data[ 222 :333 ,  0:1] 
train_targets_data_3 =  train_targets_data[ 333 :444 ,  0:1]
train_targets_data_4 =  train_targets_data[ 444 :555 ,  0:1] 

with open("input.txt", "r") as f:
 inp	= f.read()
 inp	= inp.rstrip()
 cases	= inp.split("\n")
f.close()

no_cases = len(cases)
filename_prefix  = "results"
modelname_prefix = "model"
imagename_prefix = "image"

index_list = []
mse_list = []
mae_list = []

for j in range(no_cases):
 tmp = cases[j].split(',')
 tmpstr=""
 for m in range(len(tmp)):
  tmpstr = tmpstr + tmp[m] +"_"

  filename  = filename_prefix  +"_C"+str(j)+ "_" + tmpstr
  modelname = modelname_prefix +"_C"+str(j)+ "_" + tmpstr
  imagename = imagename_prefix +"_C"+str(j)+ "_" + tmpstr
	

  print("filename=",filename)
  print("modelname=",modelname)
  train_mse_score = []
  train_mae_score = []
  val_mse_score = [] 
  val_mae_score = []
  for i in range(k):
   
   #all_scores = [] # 점수를 저장할 배열
   all_val_mse_histories = [] # 그래프값으로 띄울 걸 저장하는 배열
   loss_all_mse_histories = []
   print("="*50)
   print(j, '번째 Case, 처리중인 폴드 #', i)
   
   if(i == 0):
    best_model = ModelCheckpoint(filepath= 'model_check_0.h5', monitor = 'val_loss',save_best_only=True)
   if(i==1):
    best_model =  ModelCheckpoint(filepath= 'model_check_1.h5', monitor = 'val_loss',save_best_only=True)
   if(i==2):
    best_model =  ModelCheckpoint(filepath= 'model_check_2.h5', monitor = 'val_loss',save_best_only=True)
   if(i==3):
    best_model =  ModelCheckpoint(filepath= 'model_check_3.h5', monitor = 'val_loss',save_best_only=True)
   if(i==4):
    best_model =  ModelCheckpoint(filepath= 'model_check_4.h5', monitor = 'val_loss',save_best_only=True)
   (heights, widths) = train_data.shape
   if (i == 0) : # 각 회차마다 val_data빼고 나머지 data로 model을 학습시킴, 그 data가 pratial_train_data
    val_targets = train_targets_data_0
    val_data = train_data_0
    partial_train_data = np.concatenate((train_data_1,train_data_2,train_data_3,train_data_4),axis=0)     #val_data 를 제외하고 나머지 data셋을 합쳐서 모델을 학습한다. 
    partial_train_targets_data = np.concatenate((train_targets_data_1,train_targets_data_2,train_targets_data_3,train_targets_data_4),axis=0)
   if (i == 1) :
    val_targets = train_targets_data_1 
    val_data = train_data_1
    partial_train_data = np.concatenate((train_data_0,train_data_2,train_data_3,train_data_4),axis=0)     #val_data 를 제외하고 나머지 data셋을 합쳐서 모델을 학습한다. 
    partial_train_targets_data = np.concatenate((train_targets_data_0,train_targets_data_2,train_targets_data_3,train_targets_data_4),axis=0)
   if (i == 2) :
    val_targets = train_targets_data_2
    val_data = train_data_2
    partial_train_data = np.concatenate((train_data_1,train_data_0,train_data_3,train_data_4),axis=0)     #val_data 를 제외하고 나머지 data셋을 합쳐서 모델을 학습한다. 
    partial_train_targets_data = np.concatenate((train_targets_data_1,train_targets_data_0,train_targets_data_3,train_targets_data_4),axis=0)
   if (i == 3) :
    val_targets = train_targets_data_3
    val_data = train_data_3
    partial_train_data = np.concatenate((train_data_1,train_data_2,train_data_0,train_data_4),axis=0)     #val_data 를 제외하고 나머지 data셋을 합쳐서 모델을 학습한다. 
    partial_train_targets_data = np.concatenate((train_targets_data_1,train_targets_data_2,train_targets_data_0,train_targets_data_4),axis=0)
   if (i == 4) :
    val_targets = train_targets_data_4
    val_data = train_data_4
    partial_train_data = np.concatenate((train_data_1,train_data_2,train_data_3,train_data_0),axis=0)     #val_data 를 제외하고 나머지 data셋을 합쳐서 모델을 학습한다. 
    partial_train_targets_data = np.concatenate((train_targets_data_1,train_targets_data_2,train_targets_data_3,train_targets_data_0),axis=0)
   batch_size = 64
   model =  build_model(widths)
   history = model.fit(partial_train_data, partial_train_targets_data
             ,epochs=num_epochs, batch_size=batch_size, verbose=1
             ,validation_data = (val_data, val_targets)
             ,callbacks = [early_stopping, best_model])
   modelname2 = modelname.strip() + "k" + str(i) + ".h5"
   model.save(modelname2)
   if(i ==0):
    model_check = load_model('model_check_0.h5')
   if(i ==1):
    model_check = load_model('model_check_1.h5')
   if(i ==2):
    model_check = load_model('model_check_2.h5')
   if(i ==3):
    model_check = load_model('model_check_3.h5')
   if(i ==4):
    model_check = load_model('model_check_4.h5')

   modelname3 = modelname.strip() + "k" + str(i) + ".h5"
   model = load_model(modelname3)

   loss_mse_history = history.history['loss']
   loss_mae_history = history.history['mean_absolute_error']
   mse_val_history = history.history['val_loss'] 
   mae_val_history = history.history['val_mean_absolute_error'] 

   train_mse_score.append(loss_mse_history[-1])
   train_mae_score.append(loss_mae_history[-1])
   val_mse_score.append(mse_val_history[-1])
   val_mae_score.append(mae_val_history[-1])

   loss_all_mse_histories.append(loss_mse_history)
   all_val_mse_histories.append(mse_val_history)

   average_val_mse_history = [np.mean([x[i] for x in all_val_mse_histories]) for i in range(len(all_val_mse_histories[0]))]
   loss_average_mse_history = [np.mean([x[i] for x in loss_all_mse_histories]) for i in range(len(all_val_mse_histories[0]))]

   smooth_val_mse_history = smooth_curve(average_val_mse_history[:])
   smooth_loss_mse_history = smooth_curve(loss_average_mse_history[:])
	
   label1 = "Train MSE"
   label2 = "Validation MSE"
   plt.plot(range(1, len(smooth_loss_mse_history) +1), smooth_loss_mse_history, 'b', label=label1)
   plt.plot(range(1, len(smooth_val_mse_history) +1), smooth_val_mse_history, 'r',label=label2)
   plt.legend()
   plt.xlabel('Epochs')
   plt.ylabel('MSE')

   imagename2= imagename + "k" + str(i) + ".png"
   plt.savefig(imagename2)
   plt.close()
    

   if(i ==0 ):
    prediction_0 = model_check.predict(val_data)
    prediction_test_0 = model_check.predict(test_data)
   if(i ==1):
    prediction_1 = model_check.predict(val_data)
    prediction_test_1 = model_check.predict(test_data)
   if(i ==2):
    prediction_2 = model_check.predict(val_data)
    prediction_test_2 = model_check.predict(test_data)
   if(i ==3):
    prediction_3 = model_check.predict(val_data)
    prediction_test_3 = model_check.predict(test_data)
   if(i ==4):
    prediction_4 = model_check.predict(val_data)
    prediction_test_4 = model_check.predict(test_data)
   y = model.predict(test_data)
   filename2 = filename + "k" + str(i) + ".txt"
   with open(filename2,'w') as p:
    sum1 = 0
    sum2 = 0
    p.write(header1)
    p.write("\t")
    p.write(header2)
    p.write("\n")
    for n in range(len(y)):
     tmpPredEGR = y[n]
     a = np.array2string(tmpPredEGR)
     a = a.strip('[')
     a = a.strip(']')
     p.write(a)
     p.write("\t")
       
     tmpRealEGR = test_targets_data[n]
     b = np.array2string(tmpRealEGR)
     b = b.strip('[')
     b = b.strip(']')
     p.write(b)  
     p.write("\n")



     sum1 = sum1 + (float(a)-float(b))**2
     sum2 = sum2 + abs(float(a)-float(b))    
    mse = sum1/len(test_data)
    mae = sum2/len(test_data)


    tmpstr1 = "mse = "+ str(mse)
    mse = sum1/len(test_data)

    tmpstr2 = "mae = "+ str(mae)
    mae = sum2/len(test_data)
    p.write(tmpstr1)
    p.write("\n")
    p.write(tmpstr2)
   p.close()
   mse_list.append(mse)
   mae_list.append(mae)
   tmp = "["+str(j)+", "+str(i)+"]"
   index_list.append(tmp)
with open("summary.txt", 'w') as f:
  f.write("Test MSE LIST")
  f.write("\t")
  f.write("Test MAE LIST")
  f.write("\t")
  f.write("Train MSE LIST")
  f.write("\t")
  f.write("Train MAE LIST")
  f.write("\t")
  f.write("Val MSE LIST")
  f.write("\t")
  f.write("Val MAE LIST")
  f.write("\n")
  for i in range(len(mse_list)):
   f.write(index_list[i])
   f.write("\t")
   f.write(str(mse_list[i]))
   f.write("\t")
   f.write(str(mae_list[i]))
   f.write("\t")
   f.write(str(train_mse_score[i]))
   f.write("\t")
   f.write(str(train_mae_score[i]))
   f.write("\t")
   f.write(str(val_mse_score[i]))
   f.write("\t")
   f.write(str(val_mae_score[i]))
   f.write("\n")
  f.write("Average")
  f.write("\t")
  f.write(str(sum(mse_list)/len(mse_list)))
  f.write("\t")
  f.write(str(sum(mae_list)/len(mae_list)))
  f.write("\t")
  f.write(str(sum(train_mse_score)/len(train_mse_score)))
  f.write("\t")
  f.write(str(sum(train_mae_score)/len(train_mae_score)))
  f.write("\t")
  f.write(str(sum(val_mse_score)/len(val_mse_score)))
  f.write("\t")
  f.write(str(sum(val_mae_score)/len(val_mae_score)))
  f.write("\n")
  mean_str = "mean:"+str(mean)
  std_str  = "std:"+str(std)
  f.write(mean_str)
  f.write("\n")
  f.write(std_str)
f.close()
new_train_data = np.concatenate((prediction_0,prediction_1,prediction_2,prediction_3,prediction_4),axis=0) #예측 값들로 새로운 data를 만듬 
new_test_data = np.mean((prediction_test_0,prediction_test_1,prediction_test_2,prediction_test_3,prediction_test_4),axis=0)#예측 값은 새로 만들어진 test 
print(new_train_data.shape, train_targets_data.shape)

model_final = build_model_final(1)
build_model_final(new_train_data) #new_data로 모델만들기 

history_final = model_final.fit(new_train_data, train_targets_data
             ,epochs=num_epochs, batch_size=batch_size, verbose=1
             ,validation_data = (new_test_data, test_targets_data)
             ,callbacks = [early_stopping_1,best_model_2])

model_check_1 = load_model('model_check_6.h5')
final_prediction = model_check_1.predict(new_test_data)# 새로 만든 test_set으로 예측 

final_train_mse_score = []
final_train_mae_score = []
final_val_mse_score = []
final_val_mae_score = []

final_index_list = []
final_mse_list = []
final_mae_list = []

final_all_val_mse_histories = [] # 그래프값으로 띄울 걸 저장하는 배열
final_loss_all_mse_histories = []

final_loss_mse_history = history_final.history['loss']
final_loss_mae_history = history_final.history['mean_absolute_error']
final_mse_val_history = history_final.history['val_loss']
final_mae_val_history = history_final.history['val_mean_absolute_error']

final_train_mse_score.append(final_loss_mse_history[-1])
final_train_mae_score.append(final_loss_mae_history[-1])
final_val_mse_score.append(final_mse_val_history[-1])
final_val_mae_score.append(final_mae_val_history[-1])

final_loss_all_mse_histories.append(final_loss_mse_history)
final_all_val_mse_histories.append(final_mse_val_history)

final_average_val_mse_history = [np.mean([x[i] for x in final_all_val_mse_histories]) for i in range(len(final_all_val_mse_histories[0]))]
final_loss_average_mse_history = [np.mean([x[i] for x in final_loss_all_mse_histories]) for i in range(len(final_all_val_mse_histories[0]))]

final_smooth_val_mse_history = smooth_curve(final_average_val_mse_history[:])
final_smooth_loss_mse_history = smooth_curve(final_loss_average_mse_history[:])

label1 = "Final_Train MSE"
label2 = "Final_Validation MSE"
plt.plot(range(1, len(smooth_loss_mse_history) +1), smooth_loss_mse_history, 'b', label=label1)
plt.plot(range(1, len(smooth_val_mse_history) +1), smooth_val_mse_history, 'r',label=label2)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.savefig("last_traning.png")
plt.close()

filename2 = filename + "output" + ".txt"
with open(filename2,'w') as p:
 sum1 = 0
 sum2 = 0
 p.write(header1)
 p.write("\t")
 p.write(header2)
 p.write("\n")
 for n in range(len(final_prediction)):
  tmpPredEGR = final_prediction[n]
  a = np.array2string(tmpPredEGR)
  a = a.strip('[')
  a = a.strip(']')
  p.write(a)
  p.write("\t")
 
  tmpRealEGR = test_targets_data[n]
  b = np.array2string(tmpRealEGR)
  b = b.strip('[')
  b = b.strip(']')
  p.write(b)
  p.write("\n")  



  sum1 = sum1 + (float(a)-float(b))**2
  sum2 = sum2 + abs(float(a)-float(b))
 mse = sum1/len(new_test_data)
 mae = sum2/len(new_test_data)


 tmpstr1 = "mse = "+ str(mse)
 mse = sum1/len(new_test_data)

 tmpstr2 = "mae = "+ str(mae)
 mae = sum2/len(new_test_data)
 p.write(tmpstr1)
 p.write("\n")
 p.write(tmpstr2)
p.close()

final_mse_list.append(mse)
final_mae_list.append(mae)
tmp = "["+str(j)+", "+str(i)+"]"
final_index_list.append(tmp)
with open("last_summary.txt", 'w') as f:
 f.write("Test MSE LIST")
 f.write("\t")
 f.write("Test MAE LIST")
 f.write("\t")
 f.write("Train MSE LIST")
 f.write("\t")
 f.write("Train MAE LIST")
 f.write("\t")
 f.write("Val MSE LIST")
 f.write("\t")
 f.write("Val MAE LIST")
 f.write("\n")
 for i in range(len(final_mse_list)):
  f.write(final_index_list[i])
  f.write("\t")
  f.write(str(final_mse_list[i]))
  f.write("\t")
  f.write(str(final_mae_list[i]))
  f.write("\t")  
  f.write(str(final_train_mse_score[i]))
  f.write("\t")
  f.write(str(final_train_mae_score[i]))
  f.write("\t")
  f.write(str(final_val_mse_score[i]))
  f.write("\t")
  f.write(str(final_val_mae_score[i]))
  f.write("\n")
 f.write("Average")
 f.write("\t")
 f.write(str(sum(final_mse_list)/len(final_mse_list)))
 f.write("\t")
 f.write(str(sum(final_mae_list)/len(final_mae_list)))
 f.write("\t")
 f.write(str(sum(final_train_mse_score)/len(final_train_mse_score)))
 f.write("\t") 
 f.write(str(sum(final_val_mse_score)/len(final_val_mse_score)))
 f.write("\t")
 f.write(str(sum(final_val_mae_score)/len(final_val_mae_score)))
 f.write("\n")
 mean_str = "mean:"+str(mean)
 std_str  = "std:"+str(std)
 f.write(mean_str)
 f.write("\n")
 f.write(std_str)
f.close()


label1 = "Acc"
plt.plot(test_targets_data, final_prediction, 'b', label=label1)
plt.legend()
plt.xlabel('prediction')
plt.ylabel('y_test')

imagename3= "last_output"+ ".png"
plt.savefig("last_ouput.png")
plt.close()
