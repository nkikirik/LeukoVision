# Image-Based Classification of White Blood Cell Types

This project aims to build a deep learning model to identify different types of WBCs. 
The data that use as a referance for our model is introduced in the following publication:
https://doi.org/10.1016/j.cmpb.2019.105020
and the data is avaliable to download by the following link:
https://data.mendeley.com/datasets/snkd93bnjr/1

## What each folder do?

The folder structure is as follow:

    root/ ->contains Readme, gitignore, requirements and all subfolders below
    |_src
    |_Notebooks
        |_sampling_test
        |_modeling_test

## How to use our models?

To use our model to predict your WBCs, first, we recommend you to use our `requirements.txt` file to install all the dependant 
python packages on a new enviroment to make sure that you don't run into any technical problems. 

Second, you can download our models (h5 or pth file format) and loaded to identify your images.