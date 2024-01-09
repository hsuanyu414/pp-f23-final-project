# Canny Edge Detection (CUDA ver.)

## 使用方法

### 編譯

```
make
```

### 執行

```
./canny <input_image_path> <edge_linking_method>
```

* example
```
# using cuda edge linking
./canny ../common/data/1024.bmp cuda_bfs
# using serial edge linking
./canny ../common/data/1024.bmp 
```

### 輸出

* 執行結果分為
    1. Smoothing、Gradient Computation、Nonmaxima Suppression 和 Double Thresholding的時間加總
    2. Edge Linking的執行時間
    3. 以上兩階段的時間加總，代表整個演算法的執行時間