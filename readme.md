# requirements
If you want to run the program, please install the pytorch, pytorch-lightning, pytorchvideo first.
In our implementation, the version of pytorch-lightning is 1.9.0,  pytorchvideo is 0.1.5, torch is 1.11.0

# how to run it
1. First, you should prepare the data with a a csv file. The video is stored with images in one folder, so you should
provide the folder name in the csv file, and then provide the groundtruth for each frame following the folder path.
2. Second, configure the yml file with the data csv file and you can change the 
other parameters, such as batch size, block size, and so on.
3. 