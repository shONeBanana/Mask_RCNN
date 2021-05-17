# 
修改自以下原始碼
[[Original Codes]](https://github.com/matterport/Mask_RCNN)

## 安裝方法
1. 下載此 [程式碼](https://github.com/shONeBanana/Mask_RCNN/archive/refs/heads/master.zip) 、 [cocoapi](https://github.com/cocodataset/cocoapi) 及 coco模型檔[mask_rcnn_coco.h5
](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)   
2. 將 `Mask_RCNN-master` 和 `cocoapi-master` 個別解壓縮
3. 進入 `Mask_RCNN-master` 目錄，準備好虛擬環境，執行 `pip install -r requirments_fix.txt`。   
※如果要GPU版則改成tensorflow-gpu，CUDA 10
4. 進入 `cocoapi-master\PythonAPI`， 用文字編輯器打開 `setup.py`，將 12 行改成 `extra_compile_args=[],` ，並執行 `python setup.py install`
5. 將 `mask_rcnn_coco.h5` 放置 `Mask_RCNN-master` 目錄底下，並移動至該目錄，後續執行程式碼皆在此目錄，完成安裝。

## Requirments
修改部分套件版本號，存至 `requirments_fix.txt`，並移除 IPython[all]  
執行 `pip install -r requirments_fix.txt` 可一次安裝  
※如果要裝GPU版本，記得改成 tensorflow-gpu

numpy  
scipy  
Pillow  
cython  
matplotlib  
scikit-image==0.16.2  
tensorflow==1.14.0  
keras==2.2.5  
opencv-python==4.1.0.25  
h5py==2.10.0  
imgaug  
 
## 新增檔案
* demo.py: 利用預訓練模型測試coco資料集
* demo_train_cat.py: 訓練 coco2014 貓資料集，請將資料集放在dataset底下並保持以下資料結構:  
dataset   
　　|-coco2014   
　　　　|-test   
　　　　|-train  
　　　　|-val  
其中 train 和 val 須包含相同檔名之 .jpg 和 .json 檔案 (如需其他影像檔格式需自行修改code)  
其中 test 僅包含 .jpg 檔案 (如需其他影像檔格式需自行修改code)   
.json檔 利用 labelme 標記後即可生成  
* demo_test_cat.py: 測試 coco2014 貓資料集，留意模型路徑
* demo_shape_dataset.py: 修改自[matterport/Mask_RCNN/samples/shapes](https://github.com/matterport/Mask_RCNN/tree/master/samples/shapes)，
因為不用額外下載影像資料庫，可用於測試環境和maskrcc API是否正常

## 修改原始碼部分檔案
1. 已修改 [Mask_RCNN/mrcnn/visualize.py](https://github.com/shONeBanana/Mask_RCNN/blob/master/mrcnn/visualize.py) ，將IPython部份去掉
