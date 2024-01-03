# CUDA Canny Edge Detection

## Usage

### Build the Program

```
make
```

### Execute

```
./canny <filename> <edge_link_method>
```

```
# using cuda edge linking
./canny ../common/data/1024.bmp cuda_bfs
# using serial edge linking
./canny ../common/data/1024.bmp 
```