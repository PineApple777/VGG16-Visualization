import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def traverse_datasets(hdf_file):

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

def extract_file_perepoch(epoch, epochFolderPath, layerFolderPath):
    for num in range(1, epoch+1):
        filename = epochFolderPath + f"cifar10-weights-epoch{num:02d}.hdf5"
        filedir = layerFolderPath + f"epoch{num:02d}\\"
        if not(os.path.isdir(filedir)):
            os.makedirs(os.path.join(filedir))
        with h5py.File(filename, 'r') as f:
            for dset in traverse_datasets(filename):
                ids = dset.split('/')[3]
                textfile = filedir + f"cifar10-e{num:02d}-{dset.split('/')[2]}_{ids.split(':')[0]}.txt"
                print(textfile)
                text = open(textfile, 'a')
                text.write('Path:' + str(dset) + '\n')
                text.write('Shape:'+ str(f[dset].shape) + '\n')
                text.write('Data type:' + str(f[dset].dtype) + '\n')
                arr = f[dset][:]
                text.write(str(arr.tolist()) + '\n')
                text.close()

def extract_file_final(inputPath, outputFolder):
    if not(os.path.isdir(outputFolder)):
        os.makedirs(os.path.join(outputFolder))
    with h5py.File(inputPath, 'r') as f:
        for dset in traverse_datasets(inputPath):
            ids = dset.split('/')[3]
            textfile = outputFolder + f"cifar10-f-{dset.split('/')[2]}_{ids.split(':')[0]}.txt"
            print(textfile)
            text = open(textfile, 'w')
            text.write('Path:' + str(dset) + '\n')
            text.write('Shape:'+ str(f[dset].shape) + '\n')
            text.write('Data type:' + str(f[dset].dtype) + '\n')
            arr = f[dset][:]
            text.write(str(arr.tolist()) + '\n')
            text.close()

def analyzingData(inputPath):
    with h5py.File(inputPath, 'r') as f:
        for dset in traverse_datasets(inputPath):
            print("===========================\n")
            print('Path:' + str(dset) + '\n')
            print('Shape:'+ str(f[dset].shape) + '\n')
            print('Data type:' + str(f[dset].dtype) + '\n')
            arr_np = np.array(f[dset][:].tolist(), f[dset].dtype)
            layer_max = np.max(arr_np)
            layer_min = np.min(arr_np)
            layer_mean = np.mean(arr_np)
            layer_avg = np.average(arr_np)
            layer_var = np.var(arr_np)
            layer_std = np.std(arr_np)
            print("MaxValue (최대값): " + str(layer_max) + "\n")
            print("MinValue (최소값) : " + str(layer_min) + "\n")
            print("MeanValue (중앙값) : " + str(layer_mean) + "\n")
            print("AvgValue (평균값) : " + str(layer_avg) + "\n")
            print("VarValue (분산) : " + str(layer_var) + "\n")
            print("stdValue (표준편차) : " + str(layer_std) + "\n")
            print("===========================\n")
            new_arr = (arr_np - layer_avg) / (layer_max - layer_min)
            visualization('float16', new_arr, dset)
    return

def checkPerEpoch(startEpoch, endEpoch, perEpoch, filePath, layerName):
    if(startEpoch <= perEpoch):
        print("Cannot Access File, you must input perEpoch less than startEpoch\n")
        return
    for num in range(startEpoch, endEpoch, perEpoch):
        inputPath1 = filePath + f"cifar10-weights-epoch{num-perEpoch:02d}.hdf5"
        inputPath2 = filePath + f"cifar10-weights-epoch{num:02d}.hdf5" 
        f1 = h5py.File(inputPath1, 'r')
        f2 = h5py.File(inputPath2, 'r')
        arr1 = np.array(f1[layerName][:].tolist(), f1[layerName][:].dtype)
        arr2 = np.array(f2[layerName][:].tolist(), f2[layerName].dtype)
        grd_arr = arr2 - arr1
        # regulat data
        reg_arr = (grd_arr - np.average(grd_arr)) / (np.max(grd_arr) - np.min(grd_arr))
        print(f"{layerName.split('/')[2]} gradient {num-perEpoch:02d}-{num:02d} \n")
        print(f"max : {np.max(grd_arr)}, min : {np.min(grd_arr)}, avg : {np.average(grd_arr)}, mean : {np.mean(grd_arr)} \n")
        visualization('float32', reg_arr, f"{layerName.split('/')[2]} gradient {num-perEpoch:02d}-{num:02d}")
        f2.close()
        f1.close()

def visualization(float_type, reg_arr, title):
    plt.title(title)
    reg_arr.astype(float_type)
    plt.hist(reg_arr.reshape(-1), bins=2**8, range=(-1, 1))
    plt.show(block=False)
    plt.pause(2)
    plt.close()

if __name__ == "__main__":
    #extract_file_perepoch(250, f"D:\\extract\\perepoch\\", f"D:\\extract\\perlayer\\")
    #extract_file_final("cifar10vgg.h5", f"D:\\extract\\perlayer\\final\\")
    #analyzingData(f"D:\\extract\\perepoch\\cifar10-weights-epoch01.hdf5")
    checkPerEpoch(11, 250, 10, f"D:\\extract\\perepoch\\", "/conv2d_4/conv2d_4/kernel:0" )