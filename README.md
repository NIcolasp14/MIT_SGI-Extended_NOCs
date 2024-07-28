# MIT_SGI: Extended NOCs pipeline 
This project's goal is to extend the NOCs pipeline to create a 3D point cloud of an object given its extracted NOCs map from its 2D scene. This modified pipeline also locates and saves NOCs's last layer's embeddings. One could use this pipeline for the aforementioned cases, yet this pipeline also includes using the Farthest Point Sampling (FPS) algorithm to fragment the 3D point cloud and keep only the seed points (centers) of the fragments, into which the embeddings of the points of their corresponding fragments are pooled via averaging. So, one can download the NOCs_Pytorch pipeline found in: https://github.com/sahithchada/NOCS_PyTorch/tree/main, replace some of its files with the files of this repo, setup the environment as described below in order to:
* Create the 3D mesh of an object from the NOCs map of a 2D scene (needs to specify which object and scene)
* Downsample the 3D points using the FPS algorithm
* Pool (Average) the embeddings of the removed points into the seed point of their corresponding fragment

For a more detailed overview of the project, read the following blog post:  
[Extended NOCs](https://summergeometry.org/sgi2024/graph-based-optimal-transport-for-keypoint-matching-extended-nocs/)  

Tracking changes:
* In model.py the functions: unmold, detect, predict we modified
* The demo.py script is almost completely altered
* In utils.py the function unmold_embeddings was added 

## Project setup
#### Method 1
* conda create -n nocs python=3.8
* conda install --file requirements.txt (either using the req file from this repo or the one in the NOCs repo)
* pip install numpy
* pip install matplotlib
* pip install pycocotools
* pip install scipy
* pip install scikit-image
* conda install --yes pytorch
* conda install -c conda-forge opencv
* pip install torchvision

#### Method 2 (untested)
* conda env create -f conda_environment.yml
* pip install -r pip_requirements.txt

#### Method 3: Google Colab
If you use Google Colab, check the following file: https://colab.research.google.com/drive/1-HMHtLkjs8pYp6WlZ7pXSFLfNcxklBst?usp=sharing

## Running the Project 
* conda activate nocs
* cd /path/to/demo.py
* python demo.py
  
## Contact
For anything that comes up, do not hesitate to contact me: nicolaspigadas14@gmail.com
