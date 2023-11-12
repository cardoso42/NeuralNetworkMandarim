# Chinese Ideogram Detection with MLP

## Usage

### Training

To train an MLP (Multi-Layered Perceptron) network, simply go to the 
`src/training` directory and run the `train.py` Python script.

It depends on the files located in the 'data' directory (explained later). You 
may need to install Python libraries such as Numpy and Scikit-Learn.

### Testing Application

We have also developed a testing application that can be used to draw an 
ideogram and request the network to identify it. It can be executed by going to 
the `src/app` directory and running the `app.py` Python script.

It will open a new window where the user can interact with the neural network. 
It requires the existence of a file containing the trained network 
`trained_models/trained_mlp_model.joblib`.

## Directories

### data

Contains the images of Chinese ideograms used in our training.

It has a folder for each ideogram, named with the ideogram symbol. Each image 
file is identified by a number, ranging from 0001 to 9999.

### etc

Contains scripts and information related to the project but not essential. Here 
are the Python and Bash scripts used to process the images.

### src

Contains the main source code of the project.

#### app

Contains the source code related to the testing application.

#### training

Contains the source code related to neural network training.

### trained_models

Contains the `.joblib` files with the trained neural networks. After training, 
the file must be manually moved here. This decision was made because we may not 
want to overwrite the previous file with each new training.
