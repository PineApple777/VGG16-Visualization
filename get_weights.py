import os
import h5py
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def checkAndMakeDir(directoryPath):
    if not(os.path.isdir(directoryPath)):
        os.makedirs(os.path.join(directoryPath))

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

def getLayerNameList(inputPath):
    layerName = []
    for dset in traverse_datasets(inputPath):
        layerName.append(str(dset))
    return layerName

def extract_file_perepoch(epoch, epochFolderPath, layerFolderPath):
    for num in range(1, epoch+1):
        filename = epochFolderPath + f"cifar10-weights-epoch{num:02d}.hdf5"
        filedir = layerFolderPath + f"epoch{num:02d}\\"
        checkAndMakeDir(filedir)
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
    checkAndMakeDir(outputFolder)
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

def analyzingData(inputPath, epochNum, folderpath):
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
            #new_arr = (arr_np - layer_avg) / (layer_max - layer_min)
            ids = dset.split('/')[3]
            layer_name = f"cifar10-e{epochNum:02d}-{dset.split('/')[2]}_{ids.split(':')[0]}"
            visualization('float16', arr_np, layer_name, folderpath)
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
        grd_arr = np.abs(arr2 - arr1)
        # regulat data
        #reg_arr = (grd_arr - np.average(grd_arr)) / (np.max(grd_arr) - np.min(grd_arr))
        print(f"{layerName.split('/')[2]} gradient {num-perEpoch:02d}-{num:02d} \n")
        print(f" max : {np.max(grd_arr)}\n min : {np.min(grd_arr)}\n avg : {np.average(grd_arr)}\n mean : {np.mean(grd_arr)}\n")
        visualization('float32', grd_arr, f"{layerName.split('/')[2]} kernel gradient {num-perEpoch:02d}-{num:02d}", "per10epoch\\")
        f2.close()
        f1.close()

def totalAverageGradient(epoch, filePath, layerName):
    totalGD = []
    extra = 1.0
    value = [x for x in range(1, epoch)]
    for num in range(1, epoch):
        inputPath1 = filePath + f"cifar10-weights-epoch{num:02d}.hdf5"
        inputPath2 = filePath + f"cifar10-weights-epoch{num+1:02d}.hdf5"
        f1 = h5py.File(inputPath1, 'r')
        f2 = h5py.File(inputPath2, 'r')
        arr1 = np.array(f1[layerName][:].tolist(), f1[layerName][:].dtype)
        arr2 = np.array(f2[layerName][:].tolist(), f2[layerName].dtype)
        grd_arr = np.abs(arr2 - arr1)
        gradientValue = np.average(grd_arr)
        totalGD += [gradientValue * extra]
    totalGD = np.array(totalGD)
    fig = plt.figure()
    plt.plot(value, totalGD)
    plt.xlabel("Epoch")
    plt.ylabel("Average Gradient")
    plt.show(block=False)
    plt.pause(3)
    fig.savefig("gradientAveragePerEpoch.png")
    plt.close()

def checkConvergenceRate(epoch, filePath, outputPath):
    convergencePoint = 2 ** (-12)
    value = [x for x in range(1, epoch)]
    layers = getLayerNameList("cifar10vgg.h5")
    checkAndMakeDir(outputPath)
    for layerName in layers:
        convRate_arr = []
        print(layerName)
        for num in range(1, epoch):
            conv = 0
            inputPath1 = filePath + f"cifar10-weights-epoch{num:02d}.hdf5"
            inputPath2 = filePath + f"cifar10-weights-epoch{num+1:02d}.hdf5"
            f1 = h5py.File(inputPath1, 'r')
            f2 = h5py.File(inputPath2, 'r')
            arr1 = np.array(f1[layerName][:].tolist(), f1[layerName].dtype)
            arr2 = np.array(f2[layerName][:].tolist(), f2[layerName].dtype)
            grd_arr = np.ravel(np.abs(arr2 - arr1), order='C')
            grd_arr = grd_arr.tolist()
            length = len(grd_arr)
            for values in grd_arr:
                if values <= convergencePoint :
                    conv += 1
            convRate = (conv / length) * 100
            #print("[" + str(num) + "]" + str(convRate) + "%")
            convRate_arr.append(convRate)
        total_rate= np.array(convRate_arr)
        fig = plt.figure()
        plt.title(layerName)
        plt.plot(value, total_rate)
        plt.xlabel("Time [epoch]")
        plt.ylabel("Convergence Rate [%]")
        plt.show(block=False)
        plt.pause(3)
        ids = layerName.split('/')[3]
        fig.savefig(outputPath + f"ConvergenceRatio_{layerName.split('/')[2]}_{ids.split(':')[0]}.png")
        plt.close()

"""
def _3dGradient(epoch, filePath, layerName):
    totalGD = []
    value = [x for x in range(1, epoch)]
    for num in range(1, epoch):
        inputPath1 = filePath + f"cifar10-weights-epoch{num:02d}.hdf5"
        inputPath2 = filePath + f"cifar10-weights-epoch{num+1:02d}.hdf5"
        f1 = h5py.File(inputPath1, 'r')
        f2 = h5py.File(inputPath2, 'r')
        arr1 = np.array(f1[layerName][:].tolist(), f1[layerName][:].dtype)
        arr2 = np.array(f2[layerName][:].tolist(), f2[layerName].dtype)
        grd_arr = np.abs(arr2 - arr1)
        grd_arr = grd_arr.flatten()
        #print(grd_arr)
        totalGD += [grd_arr]
    totalGD = np.array(totalGD)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nbins = 2 ** 4
    for i in value:
        print(i)
        ys = totalGD[i-1]
        hist, bins = np.histogram(ys, bins=nbins)
        #print(hist)
        #print(bins)
        xs = (bins[:-1] + bins[1:])/2
        print(xs)
        ax.bar(xs, hist, zs=(i-1), zdir='y', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
"""

def visualization(float_type, reg_arr, title, outFolder):
    fig = plt.figure()
    plt.title(title)
    reg_arr.astype(float_type)
    plt.hist(np.ravel(reg_arr, order='C'), bins=2**8)
    plt.xlabel("Tensor Value")
    plt.ylabel("Counts")
    plt.show(block=False)
    plt.pause(1)
    path = f"D:\\extract\\analysis\\" + outFolder
    checkAndMakeDir(path)
    fig.savefig(path + title + ".png")
    plt.close()

if __name__ == "__main__":
    #extract_file_perepoch(250, f"D:\\extract\\perepoch\\", f"D:\\extract\\perlayer\\")
    #extract_file_final("cifar10vgg.h5", f"D:\\extract\\perlayer\\final\\")
    #analyzingData(f"D:\\extract\\perepoch\\cifar10-weights-epoch01.hdf5", 1, "01epoch\\")
    #analyzingData(f"cifar10vgg.h5", 250, "final_epoch\\")
    #checkPerEpoch(11, 250, 10, f"D:\\extract\\perepoch\\", "/conv2d_4/conv2d_4/kernel:0" )
    #totalAverageGradient(250, f"D:\\extract\\perepoch\\","/conv2d_4/conv2d_4/kernel:0")
    #_3dGradient(250, f"D:\\extract\\perepoch\\", "/conv2d_4/conv2d_4/kernel:0")
    checkConvergenceRate(250, f"D:\\extract\\perepoch\\", f"D:\\extract\\convergence\\")