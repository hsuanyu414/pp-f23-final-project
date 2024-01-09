# Accelerate Canny Edge Detector with Pthread

## 編譯
`make`


## 指定 # thread 
預設是使用4個thread

更改 pthread.cpp / pthread_balancing.cpp中 line 15 / 18 中的 #define THREAD_NUM

## 指定輸入圖片
預設是使用izuna24.bmp

更改 pthread.cpp / pthread_balancing.cpp中 line 358 /482 中的 filename 至欲輸入的圖片路徑即可
(輸入圖片相關規範參考 ../README.md)

## 執行
`./pthread.out`

`./pthread_balancing.out`

## 輸出
執行結果會列出各個步驟的執行時間以及其加總，並將結果輸出至 `output_with_bfs.bmp`

## 清除
`make clean`
