# 
修改自以下原始碼
[[Original Codes]](https://github.com/matterport/Mask_RCNN)
 
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
* `demo_shape_dataset.py`: 修改自[matterport/Mask_RCNN/samples/shapes](https://github.com/matterport/Mask_RCNN/tree/master/samples/shapes)，
因為不用額外下載影像資料庫，可用於測試環境和maskrcc API是否正常


## 修改原始碼部分檔案
