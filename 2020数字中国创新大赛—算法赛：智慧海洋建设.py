##############################################数据预处理###########################
import pandas as pd
file_train = pd.read_csv('C:/Users/limen/Desktop/document/competition/final_game_document/file_train.csv',index_col = 0)
file_test = pd.read_csv('C:/Users/limen/Desktop/document/competition/final_game_document/file_test.csv',index_col = 0) #这里输入新的测试文件的路径
file_test['x'] = (file_test['x'].astype('float64')).round(decimals = 3) #将科学计数法转换成 float64
file_test['y'] = (file_test['y'].astype('float64')).round(decimals = 3)
file_train_x = file_train.iloc[:,[0,1,2,3,4,5]] #提取训练集的特征列
file_total = pd.concat([file_train_x,file_test])  #将测试集和训练集纵向合并
#将'time'列,切分开,弄成'date','time'，列。
date = file_total['time'].str.split(' ',expand=True)[0]
time = file_total['time'].str.split(' ',expand=True)[1]
file_total['date'] = date
file_total['time'] = time


#将time列里的时间全部转成秒，比如   01:00:00  就是，1*60*60。
new_standard_dataset_for_time = file_total['time']
new_standard_dataset_for_time = list(new_standard_dataset_for_time)
new_standard_dataset_for_time_list = []
for i in new_standard_dataset_for_time:
    new_standard_dataset_for_time_list.append(int(i[0])*10*60*60 + int(i[1])*60*60 + int(i[3])*10*60 + int(i[4])*60 + int(i[6])*10 + int(i[7]))
file_total['time'] = new_standard_dataset_for_time_list

#全部数据转成float
file_total = file_total.astype(float)
import numpy as np
#将 file_total 的 特征集提出来，进行标准化,file_total文件包含了测试集和训练集
file_total_feature = file_total.iloc[:,[1,2,3,4,5,6]]
file_total_feature = pd.DataFrame(file_total_feature,dtype=np.float) #把date,这个object类型转成float
from sklearn import preprocessing
standard_dataset = file_total_feature.values
standard_dataset = preprocessing.scale(standard_dataset) #用sklearn进行标准化处理
standard_dataset= pd.DataFrame(standard_dataset)

new_train_dataset = standard_dataset.iloc[0:2699638,:]
new_test_dataset = standard_dataset.iloc[2699638:,:]
new_train_dataset['type'] = list(file_train['type'])

new_train_dataset.to_csv('C:/Users/limen/Desktop/document/competition/final_game_document/new_train_dataset.csv')
new_test_dataset.to_csv('C:/Users/limen/Desktop/document/competition/final_game_document/new_test_dataset.csv')

###################################开始训练###############################
# Python 3.5.1, TensorFlow 1.6.0, Keras 2.1.5
# ========================================================
# 导入模块
import os
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 读取CSV数据集，并拆分为训练集和测试集
# 该函数的传入参数为CSV_FILE_PATH: csv文件路径
def load_data(CSV_FILE_PATH):
    IRIS = pd.read_csv(CSV_FILE_PATH,index_col = 0)
    target_var = 'type'  # 目标变量
    # 数据集的特征
    features = list(IRIS.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = IRIS[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量进行编码
    IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(IRIS['target'])
    y_bin_labels = []  # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        IRIS['y' + str(i)] = transformed_labels[:, i]
    # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(IRIS[features], IRIS[y_bin_labels], \
                                                        train_size=0.999999, test_size=0.000001, random_state=0)
    return train_x, test_x, train_y, test_y, Class_dict


# 0. 开始
print("\nIris dataset using Keras/TensorFlow ")
np.random.seed(4)
tf.set_random_seed(13)

# 1. 读取CSV数据集
print("Loading Iris data into memory")
CSV_FILE_PATH = 'C:/Users/limen/Desktop/document/competition/final_game_document/new_train_dataset.csv'
train_x, test_x, train_y, test_y, Class_dict = load_data(CSV_FILE_PATH)

# 2. 定义模型
init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.Dense(units=100, input_dim=6, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=700, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=700, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=700, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=700, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=700, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=700, kernel_initializer=init, activation='relu'))


model.add(K.layers.Dense(units=3, kernel_initializer=init, activation='softmax')) #这里的3表示，输出的部分，有3个label
model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['categorical_accuracy'])


# 3. 训练模型
b_size = 500
max_epochs = 52 #这里应该是42，这里为了测试流水线，写成1
print("Starting training ")
h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")

# 5. 使用模型进行预测
file_test_for_predicted = pd.read_csv('C:/Users/limen/Desktop/document/competition/final_game_document/new_test_dataset.csv',index_col = 0)#需要被预测的数据集

species_dict = {v:k for k,v in Class_dict.items()}

unknown = file_test_for_predicted.values

predicted_result = model.predict(unknown)

predicted_result

predicted_result_for_uploading = []

count = 0
for i in np.argmax(predicted_result,axis = 1):
    predicted_result_for_uploading.append(species_dict[i])
    count = count + 1
    print(count)
predicted_result_for_uploading_DataFrame = pd.DataFrame(predicted_result_for_uploading)
predicted_result_for_uploading_DataFrame['渔船ID'] = list(file_test['渔船ID'])

predicted_result_for_uploading_DataFrame

#把上面生成的渔船ID剔除掉
predicted_result_for_uploading_DataFrame.drop(columns=['渔船ID'],inplace=True)

daichuli_2 = predicted_result_for_uploading_DataFrame.iloc[:,0].values.tolist() #dataframe转成list

###############################最后的处理#######################################
from collections import Counter
counter = 0 #只是计数要从0开始啊
final_result_list_for_uploading = []
right_prediction = []
t_predicted_result_list = daichuli_2
ID_of_final_file = [x for x in range(9000,11000)]  #ID是从7000开始，但是计数要从0开始啊
for i in ID_of_final_file:
    tem_list= []
    length = len(file_test[file_test['渔船ID']==i])
    for j in t_predicted_result_list[counter:counter + length]:
        tem_list.append(j)
    if (Counter(tem_list).most_common(1)[0][1])/len(tem_list) >= 0.60 : #这个数字至关重要，0.62是一个估计的大概数字
        final_result_list_for_uploading.append(i) #预测对了的ID
        right_prediction.append(Counter(tem_list).most_common(1)[0][0])
    counter = counter + length


######################################做最后的结果############################################
problem = [i for i in ID_of_final_file if i not in final_result_list_for_uploading]

correction_prediction = []
for t in problem:
    d = Counter(list(predicted_result_for_uploading_DataFrame[predicted_result_for_uploading_DataFrame['渔船ID']==t][0]))
    kk = sorted(d.items(),key = lambda d:d[1],reverse=True)[1][0]
    correction_prediction.append(kk)

count1 = 0
count2 = 0
final_final_result = []
for x in range(9000,11000):
    if x in final_result_list_for_uploading:
        final_final_result.append(right_prediction[count1])
        count1 = count1 + 1
    else:
        final_final_result.append(correction_prediction[count2])
        count2 = count2 + 1
        
final_final_result_dataframe = pd.DataFrame(final_final_result)
final_final_result_dataframe.to_csv('C:/Users/limen/Desktop/document/competition/final_game_document/result.csv',encoding = 'utf_8_sig')