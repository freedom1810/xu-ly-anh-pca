import numpy as np
from sklearn.decomposition import PCA
from read_data import *
from shutil import copyfile

if __name__ == "__main__":

    size = (128, 128)
    n_components = 100

    dir_, data = load_data(size = size)

    pca = PCA(n_components= n_components)
    pca.fit(data)
    data_pca = pca.transform(data)

    img = cv2.imread('test.jpg')
    img = cv2.resize(img, size).flatten().astype(np.float64)
    img_pca = pca.transform(np.array([img]))


    res = []
    for i in range(len(data_pca)):
        
        dis = np.linalg.norm(img_pca - data_pca[i])
        res.append([dis, dir_[i]])

    res = sorted(res, key=lambda x: x[0])

    for i in range(5):
        copyfile(dir_[i], 'res/pca-sklearn-top{}.jpg'.format(i+1))