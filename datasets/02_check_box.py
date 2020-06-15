import xml.etree.ElementTree as xml_tree
import pandas as pd
import numpy as np
import os
import shutil


def check_box(path):
    files = os.listdir(path)
    i = 0
    for anna_file in files:
        tree = xml_tree.parse(path+"/"+anna_file)
        root = tree.getroot()

        # Image shape.
        size = root.find('size')
        shape = [int(size.find('height').text),
               int(size.find('width').text),
               int(size.find('depth').text)]
        # Find annotations.
        bboxes = []
        labels = []
        labels_text = []
        difficult = []
        truncated = []
        
        for obj in root.findall('object'):
            # label = obj.find('name').text
            # labels.append(int(dataset_common.VOC_LABELS[label][0]))
            # # labels_text.append(label.encode('ascii'))
            # labels_text.append(label.encode('utf-8'))


            # isdifficult = obj.find('difficult')
            # if isdifficult is not None:
            #     difficult.append(int(isdifficult.text))
            # else:
            #     difficult.append(0)

            # istruncated = obj.find('truncated')
            # if istruncated is not None:
            #     truncated.append(int(istruncated.text))
            # else:
            #     truncated.append(0)

            bbox = obj.find('bndbox')
            # bboxes.append((float(bbox.find('ymin').text) / shape[0],
            #              float(bbox.find('xmin').text) / shape[1],
            #              float(bbox.find('ymax').text) / shape[0],
            #              float(bbox.find('xmax').text) / shape[1]
            #              ))
            if (float(bbox.find('ymin').text) >= float(bbox.find('ymax').text)) or (float(bbox.find('xmin').text) >= float(bbox.find('xmax').text)):
                print(anna_file)
                i += 1
                try:
                    shutil.move(path+"/"+anna_file,"./error2/"+anna_file)
                    shutil.move("./myData/JPEGImages/"+anna_file.split(".")[0]+".jpg","./error2/"+anna_file.split(".")[0]+".jpg")
                except:
                    pass

    print(i)



if __name__ == "__main__":
    check_box("./myData/Annotations")