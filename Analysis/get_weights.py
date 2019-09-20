import os
import h5py
from analysisUtilities import utilities
import numpy as np
import matplotlib.pyplot as plt

class AnalysisWithHdf5:
    def __init__(self, maxEpoch, inputPath):
        self.tools = utilities()
        self.maxEpoch = maxEpoch
        self.inputPath = inputPath
    
    def extract_file_perepoch(self, layerFolderPath):
        for num in range(1, self.maxEpoch+1):
            filename = self.inputPath + f"cifar10-weights-epoch{num:02d}.hdf5"
            filedir = layerFolderPath + f"epoch{num:02d}\\"
            self.tools.checkAndMakeDir(filedir)
            with h5py.File(filename, 'r') as f:
                for dset in self.tools.traverse_datasets(filename):
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

    def extractOneEpoch(self, fileName, outputFolder):
        inputFile = self.inputPath + fileName
        self.tools.checkAndMakeDir(outputFolder)
        with h5py.File(inputFile, 'r') as f:
            for dset in self.tools.traverse_datasets(inputFile):
                ids = dset.split('/')[3]
                textfile = outputFolder + f"cifar10-{dset.split('/')[2]}_{ids.split(':')[0]}.txt"
                print(textfile)
                text = open(textfile, 'w')
                text.write('Path:' + str(dset) + '\n')
                text.write('Shape:'+ str(f[dset].shape) + '\n')
                text.write('Data type:' + str(f[dset].dtype) + '\n')
                arr = f[dset][:]
                text.write(str(arr.tolist()) + '\n')
                text.close()

    def analyzingData(self, inputFile, epochName):
        filePath = self.inputPath + inputFile
        with h5py.File(filePath, 'r') as f:
            for dset in self.tools.traverse_datasets(filePath):
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
                layer_name = f"cifar10-{epochName}-{dset.split('/')[2]}_{ids.split(':')[0]}"
                self.tools.visualization('float16', arr_np, layer_name, epochName + "//")
        return

    def HeatmapData(self, fileName):
        filePath = self.inputPath + fileName
        layers = self.tools.getLayerNameList(filePath)
        print("Layer Counts : " + str(len(layers)))
        for x in range(len(layers)):
            f1 = h5py.File(filePath, 'r')
            if(len(f1[layers[x]].shape) != 4 ):
                continue
            arr1 = np.array(f1[layers[x]][:].tolist(),'float32')
            height = f1[layers[x]].shape[2]
            width = f1[layers[x]].shape[3]
            h_filter = f1[layers[x]].shape[0]
            w_filter = f1[layers[x]].shape[1]
            arr = np.arange(h_filter *height *w_filter *width).reshape((h_filter *height, w_filter *width)).astype('float32')
            for i in range(h_filter):
                for j in range(w_filter):
                    for a in range(height):
                        for b in range(width):
                            arr[height *i + a][width *j + b] = arr1[i][j][a][b] 
            print("Layer Name : " + layers[x])
            layerName = layers[x].split(":")[0].split("/")[1]
            layerIdf = layers[x].split(":")[0].split("/")[3]
            self.tools.visualHeatmap('float32', arr, layerName + "-" + layerIdf )
            f1.close()


    def checkPerEpoch(self, startEpoch, perEpoch, layerName):
        if(startEpoch <= perEpoch):
            print("Cannot Access File, you must input perEpoch less than startEpoch\n")
            return
        for num in range(startEpoch, self.maxEpoch, perEpoch):
            inputPath1 = self.inputPath + f"cifar10-weights-epoch{num-perEpoch:02d}.hdf5"
            inputPath2 = self.inputPath + f"cifar10-weights-epoch{num:02d}.hdf5"
            f1 = h5py.File(inputPath1, 'r')
            f2 = h5py.File(inputPath2, 'r')
            arr1 = np.array(f1[layerName][:].tolist(), f1[layerName][:].dtype)
            arr2 = np.array(f2[layerName][:].tolist(), f2[layerName].dtype)
            grd_arr = np.abs(arr2 - arr1)
            # Regulate Data
            #reg_arr = (grd_arr - np.average(grd_arr)) / (np.max(grd_arr) - np.min(grd_arr))
            print(f"{layerName.split('/')[2]} gradient {num-perEpoch:02d}-{num:02d} \n")
            print(f" max : {np.max(grd_arr)}\n min : {np.min(grd_arr)}\n avg : {np.average(grd_arr)}\n mean : {np.mean(grd_arr)}\n")
            self.tools.visualization('float32', grd_arr, f"{layerName.split('/')[2]} kernel gradient {num-perEpoch:02d}-{num:02d}", "per10epoch\\")
            f2.close()
            f1.close()

    def totalAverageGradient(self, layerName):
        totalGD = []
        extra = 1.0
        value = [x for x in range(1, self.maxEpoch)]
        for num in range(1, self.maxEpoch):
            inputPath1 = self.inputPath + f"cifar10-weights-epoch{num:02d}.hdf5"
            inputPath2 = self.inputPath + f"cifar10-weights-epoch{num+1:02d}.hdf5"
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

    def checkConvergenceRate(self, inputFile, outputPath):
        convergencePoint = 2 ** (-12)
        inputP = self.inputPath + inputFile
        value = [x for x in range(1, self.maxEpoch)]
        layers = self.tools.getLayerNameList(inputP)
        self.tools.checkAndMakeDir(outputPath)
        for layerName in layers:
            convRate_arr = []
            print(layerName)
            for num in range(1, self.maxEpoch):
                conv = 0
                inputPath1 = self.inputPath + f"cifar10-weights-epoch{num:02d}.hdf5"
                inputPath2 = self.inputPath + f"cifar10-weights-epoch{num+1:02d}.hdf5"
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


# Example of Running VGG16 Visualization
if __name__ == "__main__":
    myAnalysis = AnalysisWithHdf5(250, f"H:\\extract\\perepoch\\")
    #myAnalysis.extract_file_perepoch(f"H:\\extract\\perlayer\\")
    #myAnalysis.extractOneEpoch(f"cifar10-weights-epoch01.hdf5", f"H:\\extract\\perlayer\\oneEpoch\\")
    #myAnalysis.analyzingData(f"cifar10-weights-epoch01.hdf5", "epoch01")
    #myAnalysis.analyzingData(f"cifar10-weights-epoch250.hdf5", "epoch250")
    myAnalysis.HeatmapData(f"cifar10-weights-epoch250.hdf5")
    #myAnalysis.checkPerEpoch(11, 10, "/conv2d_4/conv2d_4/kernel:0")
    #myAnalysis.totalAverageGradient("/conv2d_4/conv2d_4/kernel:0")
    #myAnalysis.checkConvergenceRate(f"cifar10-weights-epoch250.hdf5", f"H:\\extract\\convergence\\")

    