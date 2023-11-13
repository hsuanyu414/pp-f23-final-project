import cv2
import numpy as np

im = cv2.imread('test_gray.bmp', cv2.IMREAD_GRAYSCALE)

# conv 
gauss33_approx = np.array(
    [
        [1, 2, 1], 
        [2, 4, 2], 
        [1, 2, 1]]
        ) / 16

after_conv = cv2.filter2D(im, -1, gauss33_approx)
#   x >>>
# y
# v
# v

for i in range(0, 6):
    for j in range(0, 6):
        print(im[i][j], end=' ')
    print()
    
print('----------------------')

for i in range(0, 6):
    for j in range(0, 6):
        print(after_conv[i][j], end=' ')
    print()


cv2.imwrite('test_gray_output_cv2.bmp', after_conv)