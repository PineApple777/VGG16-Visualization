import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

class utilities:
    def __init__(self):
        self.name = "utilties"
        self.hostDir = "H://extract//"

    def checkAndMakeDir(self, directoryPath):
        if not(os.path.isdir(directoryPath)):
            os.makedirs(os.path.join(directoryPath))

    def traverse_datasets(self, hdf_file):
        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = f'{prefix}/{key}'
                if isinstance(item, h5py.Dataset): # test for dataset
                    yield (path, item)
                elif isinstance(item, h5py.Group): # test for group (go down)
                    yield from h5py_dataset_iterator(item, path)
        with h5py.File(hdf_file, 'r') as f:
            for path, _ in h5py_dataset_iterator(f):
                yield path

    def getLayerNameList(self, inputPath):
        layerName = []
        for dset in self.traverse_datasets(inputPath):
            layerName.append(str(dset))
        return layerName

    def visualization(self, float_type, reg_arr, title, outFolder):
        fig = plt.figure()
        plt.title(title)
        reg_arr.astype(float_type)
        plt.hist(np.ravel(reg_arr, order='C'), bins=2**8)
        plt.xlabel("Tensor Value")
        plt.ylabel("Counts")
        plt.show(block=False)
        plt.pause(1)
        path = self.hostDir + f"analysis\\" + outFolder
        self.checkAndMakeDir(path)
        fig.savefig(path + title + ".png")
        plt.close()

    def visualHeatmap(self, float_type, reg_arr, title):
        fig = plt.figure()
        plt.title(title)
        reg_arr.astype(float_type)
        plt.imshow(reg_arr, cmap='hot', interpolation='nearest')
        plt.show(block=False)
        plt.pause(1)
        path = self.hostDir + f"heatmap\\"
        self.checkAndMakeDir(path)
        fig.savefig(path + title + ".png")
        plt.close()
