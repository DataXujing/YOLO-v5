import cv2
import os
import shutil

def check_img(img_path):
    imgs = os.listdir(img_path)
    for img in imgs:
        if img.split(".")[-1] !="jpg":
            print(img)
            shutil.move(img_path+"/"+img,"./error/"+img)

def check_anno(anno_path):
    anno_files = os.listdir(anno_path)
    for file in anno_files:
        if file.split(".")[-1] !="xml":
            print(file)
            shutil.move(anno_path+"/"+file,"./error/"+file)

def ckeck_img_label(img_path,anno_path):
    imgs = os.listdir(img_path)
    anno_files = os.listdir(anno_path)

    files = [i.split(".")[0] for i in anno_files]


    for img in imgs:
        if img.split(".")[0] not in files:
            print(img)
            shutil.move(img_path+"/"+img,"./error/"+img)

    imgs = os.listdir(img_path)
    images = [j.split(".")[0] for j in imgs]

    for file in anno_files:
        if file.split(".")[0] not in images:
            print(file)
            shutil.move(anno_path+"/"+file,"./error/"+file)


if __name__ == "__main__":
    img_path = "./myData/JPEGImages"
    anno_path = "./myData/Annotations"
    
    print("============check image=========")
    check_img(img_path)

    print("============check anno==========")
    check_anno(anno_path)
    print("============check both==========")
    ckeck_img_label(img_path,anno_path)
    