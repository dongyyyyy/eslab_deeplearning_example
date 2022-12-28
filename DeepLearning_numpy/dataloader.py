import numpy as np

def dataLoader(data, label, batchsize):
    # Shuffle the data
    s_idx = np.arange(data.shape[0])
    np.random.shuffle(s_idx)
    data = data[s_idx]
    label = label[s_idx]

    # Reshape the data with batch size
    data = data.reshape(int(data.shape[0] / batchsize), batchsize, -1)
    label = label.reshape(int(label.shape[0] / batchsize), batchsize, -1)

    batches = [list(i) for i in zip(data, label)]

    return iter(batches)

