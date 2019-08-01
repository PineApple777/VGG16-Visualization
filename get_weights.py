import os
import h5py
import numpy as np

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


for num in range(1, 250):
    filename = f"D:\\extract\\perepoch\\cifar10-weights-epoch{num:02d}.hdf5"
    if not(os.path.isdir(f"D:\\extract\\perlayer\\epoch{num:02d}")):
        os.makedirs(os.path.join(f"D:\\extract\\perlayer\\epoch{num:02d}"))
    with h5py.File(filename, 'r') as f:
        for dset in traverse_datasets(filename):
            ids = dset.split('/')[3]
            textfile = f"D:\\extract\\perlayer\\epoch{num:02d}\\cifar10-e{num:02d}-{dset.split('/')[2]}_{ids.split(':')[0]}.txt"
            print(textfile)
            text = open(textfile, 'a')
            text.write('Path:' + str(dset) + '\n')
            text.write('Shape:'+ str(f[dset].shape) + '\n')
            text.write('Data type:' + str(f[dset].dtype) + '\n')
            arr = f[dset][:]
            text.write(str(arr.tolist()) + '\n')
            text.close()


finalPath = "cifar10vgg.h5"
if not(os.path.isdir(f"D:\\extract\\perlayer\\final")):
    os.makedirs(os.path.join(f"D:\\extract\\perlayer\\final"))
with h5py.File(finalPath, 'r') as f:
    for dset in traverse_datasets(finalPath):
        ids = dset.split('/')[3]
        textfile = f"D:\\extract\\perlayer\\final\\cifar10-f-{dset.split('/')[2]}_{ids.split(':')[0]}.txt"
        print(textfile)
        text = open(textfile, 'w')
        text.write('Path:' + str(dset) + '\n')
        text.write('Shape:'+ str(f[dset].shape) + '\n')
        text.write('Data type:' + str(f[dset].dtype) + '\n')
        arr = f[dset][:]
        text.write(str(arr.tolist()) + '\n')
        text.close()
