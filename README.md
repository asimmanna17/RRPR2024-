# # Source codes for paper-  Learning Neural Networks for Multi-label Medical Image Retrieval Using Hamming Distance Fabricated with Jaccard Similarity Coefficient\\ JaccHash: Multi-label Medical Image Retrieval Using Hamming Distance Between Hash Codes and Jaccard Similarity Coefficient Between Image Label Sets


## Prerequisites
* Ubuntu\* 20.04
* Debian
* Python\* 3.9.6
* NVidia\* GPU for training
* 16GB RAM for inference
* CUDA 11.2
  
## Setup
* Install Git LFS: `
* Create enviorment: `conda create -n rrpr python=3.9.6 pip`
* Activate enviorment: `source activate rrpr`
* Download: `git clone https://github.com/asimmanna17/RRPR2024.git`
* Change directory: `cd RRPR2024`
* pip install: `pip install -r requirements.txt`

## Instruction
Please ensure you read the following guidelines before running the code. Additionally, run `check_package.py` to verify that all the required packages and libraries are available in your environment.

## Used datasets
The dataset is sourced from the publicly available NIH Chest X-ray database , which contains 112,120 frontal-view X-ray images from 30,805 unique patients[[1]](#nihdataset). From this dataset, we selected 51,480 images. These images are organized into three distinct sets: a training set with 38,610 images, a gallery set with 10,296 images, and a query set with 2,574 images. All images are stored in `.npy` format. The training set is used during training, while the gallery and query sets are used during inference. The dataset is available at: \url{https://data.mendeley.com/datasets/c5x35tmj5v/1}. After downloading and extracting the 'Dataset.zip' file, three image subfolders are provided: 'train', 'gallery', and 'query'. Note that a subset of sample dataset is already availabe in `./Dataset` folder. 

## Implementation guidelines 
Codes can be implemented for two purposes: code verification and result reproducibility.
#### Code verification:
In this case, the code for the algorithm is demonstrated using a very small subset availabe in `./Dataset` folder. 
#### Result reproducibility:
To reproduce the results shown in **Table 2 and Table 3**, the full dataset provided above must be used and saved in the appropriate directory, i.e., `.\Dataset`. For training, the `train.py` script should be executed using the `train` folder by providing hash code length {16,32,48,64}. **However, the training process can be skipped since the pre-trained model has already been uploaded at `.\Datastore\Models`**. Once the trained model is available, the inference results can be reproduced by running `evaluation.py` with the `gallery` and '`query` folders. Ensure that all data paths are correctly linked to the code.

\noindent **For Table 4 and Figure 5**, the notebook `demo.ipynb` can be used.

## **Contributor**

The codes/model is contributed  by

<a href="https://www.linkedin.com/in/asimmanna17/">Asim Manna</a>, </br>
Department of Artificial Intelligence, </br>
Indian Institute of Technology Kharagpur </br>
email: asimmanna17@kgpian.iitkgp.ac.in </br> 

## **References**

<div id="nihdataset">
<a href=#>[1] </a>Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M. and Summers, R.M., 2017. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2097-2106).
</dice>
