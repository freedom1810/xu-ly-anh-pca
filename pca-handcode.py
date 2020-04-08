from read_data import *
from scipy import linalg
from shutil import copyfile


def svd_flip(u, v):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]

    return u, v


def pca(X = None, n_components = 100):
    
    mean_ = np.mean(X, axis=0)
    X -= mean_
    U, S, V = linalg.svd(X, full_matrices=False)
    U, V = svd_flip(U, V)

    components_ = V
    components_ = components_[:n_components]
    return mean_, components_


def pca_transform(X = None, mean = None, components = None):

    X = X - mean
    X_transformed = np.dot(X, components.T)
    # if self.whiten:
    #     X_transformed /= np.sqrt(explained_variance_)
    return X_transformed


if __name__ == "__main__":
    size = (128, 128)
    n_components = 100

    dir_, data = load_data(size = size)

    mean, components = pca(X = np.copy(data), n_components = n_components)

    data_pca = pca_transform(X = np.copy(data), mean = mean, components = components)

    img = cv2.imread('test.jpg')
    img = cv2.resize(img, size).flatten().astype(np.float64)
    img_pca = pca_transform(X = np.array(img), mean = mean, components = components)


    res = []
    for i in range(len(data_pca)):
        
        dis = np.linalg.norm(img_pca - data_pca[i])
        res.append([dis, dir_[i]])

    res = sorted(res, key=lambda x: x[0])

    for i in range(5):
        copyfile(dir_[i], 'res/pca-hand-code-top{}.jpg'.format(i+1))


    






