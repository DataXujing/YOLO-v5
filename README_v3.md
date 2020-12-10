### YOLO v5转TensorRT模型并调用

### 0.pt模型转wts模型

```
python3 gen_wts.py
# 注意修改代码中模型保存和模型加载的路径
```



### 1.修改部分文件

+ 0.修改CMakeLists.txt

```
cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt <------------------
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(/home/myuser/xujing/TensorRT-7.0.0.11/)
link_directories(/home/myuser/xujing/TensorRT-7.0.0.11/)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)


```



+ 1.把tensorRT安装包下的bin文件的内容copy到yolov5文件夹
![](pic/p1.png)
+ 2.修改yololayer.h

```c++
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;  //20000
    static constexpr int CLASS_NUM = 17;  //需要修改
    static constexpr int INPUT_H = 640;  //需要修改
    static constexpr int INPUT_W = 640;  //需要修改
```



+ 3.修改yolov5.cpp

```c++
#define NET x  // s m l x  修改网络类型,我们用的是x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// #define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id 
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25
#define BATCH_SIZE 1
```



### 2.编译YOLOv5

```shell
1. generate yolov5s.wts from pytorch with yolov5s.pt, or download .wts from model zoo

git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
// download its weights 'yolov5s.pt'
// copy tensorrtx/yolov5/gen_wts.py into ultralytics/yolov5
// ensure the file name is yolov5s.pt and yolov5s.wts in gen_wts.py
// go to ultralytics/yolov5
python gen_wts.py
// a file 'yolov5s.wts' will be generated.

2. build tensorrtx/yolov5 and run

// put yolov5s.wts into tensorrtx/yolov5
// go to tensorrtx/yolov5
// ensure the macro NET in yolov5.cpp is s 
mkdir build
cd build
cmake ..
make
```

### 3.序列化引擎

```shell
sudo ./yolov5 -s             // serialize model to plan file i.e. 'yolov5s.engine'
# 测试序列化的引擎是否可用
sudo ./yolov5 -d  ./sample // deserialize plan file and run inference, the images in samples will be processed.
```



### 4.YOLO v5 tensorRT加速Python调用

```
python3 yolov5_trt.py
# 注意修改模型类别，序列化引擎加载的路径，测试图像的路径
```





#### Reference

https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5

https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/run_on_windows.md