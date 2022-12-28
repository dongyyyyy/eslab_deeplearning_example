import numpy as np

def one_hot(y, length_of_onehot):
    return np.eye(length_of_onehot)[y].reshape(-1, length_of_onehot)


def crossentropy(z, y):  # sample의 평균 // 클래스간의 합
    return np.mean(-1 * np.sum((one_hot(y,10) * np.log(z)), axis=1))


def ave_accuracy(output, y):
    return np.mean(np.argmax(output, axis=1).reshape(-1, 1) == y, axis=0)