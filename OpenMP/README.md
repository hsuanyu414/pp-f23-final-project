# Canny Edge Detection (OpenMP ver.)

## 檔案說明
- `omp.cpp` : 使用 OpenMP 平行化的版本
- `omp_visit_lock.cpp` : omp.cpp 在 edge linking 階段使用 visit lock 的版本

## 編譯
`make`

## 指定輸入圖片
更改 
- omp.cpp 中 line 273
- omp_visit_lock.cpp 中 line 288 
中的 filename 至欲輸入的圖片路徑即可
(輸入圖片相關規範參考 ../README.md)

## 執行
`./omp <thread_num>`
`./omp_visit_lock <thread_num>`

## 實驗調整
若要進行 omp 及 omp_visit_lock 的實驗，請取消 omp.cpp line 225 及 line 265~268 的註解

## 輸出
執行結果會列出各個步驟的執行時間以及其加總，並將結果輸出至 `output.bmp` 中
實驗部分還有列出 edge linking 階段總共拜訪的 pixel 數量

## 清除
`make clean`
