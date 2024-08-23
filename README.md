# SRGAN Implementation using Pytorch

This repository contains an implementation of the **Image Super-Resolution Using Deep Convolutional Networks** paper by Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, et al. The implementation is done using PyTorch and includes a Jupyter notebook demonstrating the training and evaluation of the GAN model for image super-resolution.

For more detail about how SRGAN works, you can check out my blog post series on dev.to [here](https://dev.to/adamazuddin/series/27510)

## Contents

- `data/`: Contains the images used for training and testing
- `results/`: Contains the result of test images when inferenced with the model
- `SRGAN_train.ipynb`: Jupyter notebook containing code for training and evaluating the SRCNN model.
- `requirements.txt`: File listing the required Python packages for running the code.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AdamAzuddin/SRGAN-PyTorch.git
   cd SRGAN-PyTorch
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `SRGAN_train.ipynb` in Jupyter Notebook to train the SRGAN model.
4. After training, use the trained model to perform image super-resolution on new images.

## References

- Original Paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802)
- PyTorch: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
