import os
import random
 
trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
lists = range(num)

tr = int(num * train_percent)
train = random.sample(lists, tr)
 

ftrain = open('./ImageSets/Main/train.txt', 'w')
fval = open('./ImageSets/Main/val.txt', 'w')
 
for i in lists:
    name = total_xml[i][:-4] + '\n'
    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)

 

ftrain.close()
fval.close()

# voc Main/train,val 图像名生成