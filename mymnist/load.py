import struct
import os


class MNIST:

    def __init__(self):
        self.data_path = '/home/lathrom-g/Projects/pytorch-tutorial/data/'
        self.load_training_labels()
        self.load_training_images()
        self.load_testing_labels()
        self.load_testing_images()



    def load_training_labels(self):
        fn = 'train-labels-idx1-ubyte'
        print(f'\n\nTraining Labels File: {fn}')
        with open(self.data_path+fn, 'rb') as fp:
            chunk = fp.read(2*4)

            
            magic, numLabels = struct.unpack(">ii", chunk)
            if magic == 2049:
                print(f'Magic Number: 0x{magic:08x} = {magic} (OK)')
            else:
                raise DataError("Magic number does not match on training label file!")

            if numLabels == 60000:
                print(f'Number of Labels: {numLabels} (OK)')
            else:
                raise DataError("Number of labels in training label file incorrect!")

            chunk = fp.read(numLabels)

            self.training_labels = list()
            for label in struct.iter_unpack(">B", chunk):
                self.training_labels.append(label[0])

            print('First 10 Labels: ', self.training_labels[:10])


    def load_training_images(self):
        fn = 'train-images-idx3-ubyte'
        print(f'\n\nTraining Images File: {fn}')
        with open(self.data_path+fn, 'rb') as fp:
            chunk = fp.read(4*4)

            
            magic, numImages, self.image_rows, self.image_colums = struct.unpack(">iiii", chunk)
            if magic == 2051:
                print(f'Magic Number: 0x{magic:08x} = {magic} (OK)')
            else:
                raise DataError("Magic number does not match on training image file!")

            if numImages == 60000:
                print(f'Number of Images: {numImages} (OK)')
            else:
                raise DataError("Number of images in training file incorrect!")
            
            if self.image_rows == 28:
                print(f'Number Rows in image: {self.image_rows} (OK)')
            else:
                raise DataError("Number of rows in image in training file incorrect!")

            if self.image_colums == 28:
                print(f'Number of columns in image: {self.image_colums} (OK)')
            else:
                raise DataError("Number of columns in image in training file incorrect!")
            
            chunk = fp.read(numImages*self.image_rows*self.image_colums) 

            self.testing_values = list()
            for label in struct.iter_unpack(">B", chunk):
                self.testing_values.append(label[0])

            print('First 10 character values: ', self.testing_values[:10])

            if len(self.testing_values) == numImages*self.image_rows*self.image_colums:
                print(f'Number of items read correct')
            else:
                raise DataError(f'Number of items read doens\'t match number needed for {fn}')

    def load_testing_labels(self):
        fn = 't10k-labels-idx1-ubyte'
        print(f'\n\nTesting Labels File: {fn}')
        with open(self.data_path+fn, 'rb') as fp:
            chunk = fp.read(2*4)

            
            magic, numLabels = struct.unpack(">ii", chunk)
            if magic == 2049:
                print(f'Magic Number: 0x{magic:08x} = {magic} (OK)')
            else:
                raise DataError("Magic number does not match on testing label file!")

            if numLabels == 10000:
                print(f'Number of Labels: {numLabels} (OK)')
            else:
                raise DataError("Number of labels in testing label file incorrect!")

            chunk = fp.read(numLabels)

            self.testing_labels = list()
            for label in struct.iter_unpack(">B", chunk):
                self.testing_labels.append(label[0])

            print('First 10 Labels: ', self.testing_labels[:10])


    def load_testing_images(self):
        fn = 't10k-images-idx3-ubyte'
        print(f'\n\nTesting Images File: {fn}')
        with open(self.data_path+fn, 'rb') as fp:
            chunk = fp.read(4*4)

            
            magic, numImages, self.image_rows, self.image_colums = struct.unpack(">iiii", chunk)
            if magic == 2051:
                print(f'Magic Number: 0x{magic:08x} = {magic} (OK)')
            else:
                raise DataError("Magic number does not match on testing image file!")

            if numImages == 10000:
                print(f'Number of Images: {numImages} (OK)')
            else:
                raise DataError("Number of images in testing file incorrect!")
            
            if self.image_rows == 28:
                print(f'Number Rows in image: {self.image_rows} (OK)')
            else:
                raise DataError("Number of rows in image in testing file incorrect!")

            if self.image_colums == 28:
                print(f'Number of columns in image: {self.image_colums} (OK)')
            else:
                raise DataError("Number of columns in image in testing file incorrect!")
            
            chunk = fp.read(numImages*self.image_rows*self.image_colums) 

            self.training_values = list()
            for label in struct.iter_unpack(">B", chunk):
                self.training_values.append(label[0])

            print('First 10 character values: ', self.training_values[:10])

            if len(self.training_values) == numImages*self.image_rows*self.image_colums:
                print(f'Number of items read correct')
            else:
                raise DataError(f'Number of items read doens\'t match number needed for {fn}')

if __name__ == '__main__':
    mnist = MNIST()

