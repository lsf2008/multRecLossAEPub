This is a PyTorch-lightning/GPU implementation of the paper 
Video anomaly detection based on a multi-layer
reconstruction autoencoder with a variance attention
strategy.

# requirements
To run the program, please install the following packages: pytorch, pytorch-lightning, and pytorchvideo. In our implementation, we used pytorch-lightning version 1.9.0, pytorchvideo 
version 0.1.5, and torch version 1.11.0.


# how to run it`
1. To start, prepare the data using a CSV file. Since the video is stored with images in a single folder, provide the folder name in the CSV file followed by the ground truth for each frame.
2. Next, configure the YAML file using the data CSV file, and adjust other parameters such as batch size, learning rate, weight decay, and block size, etc.
3. Finally, specify the "flg" in the [tst.py](http://tst.py/) file and run it.