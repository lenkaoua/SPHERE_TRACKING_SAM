import matplotlib.pyplot as plt

CV_array = [[1, 2, 3], [4, 5, 6],[7, 8, 9], [1, 2, 0]]

CV_array.insert(0,[1]*3)

plt.imshow(CV_array, cmap=plt.cm.Greys_r)

plt.show()