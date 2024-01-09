# Canny Edge Detection (CUDA ver.)

## 使用方法

### 編譯

```
make
```

### 執行

```
./canny <filename> <edge_linking_method>
```

* example
```
# using cuda edge linking
./canny ../common/data/1024.bmp cuda_bfs
# using serial edge linking
./canny ../common/data/1024.bmp 
```