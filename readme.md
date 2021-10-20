# Face Landmarks Generation Using 3DDFA_V2

This project can be used for generating face landmarks data using 3DDFA_V2 library and 
storing it to a csv and generate an annoated video with the keypoints plotted.


# About MediaPipe Face Mesh 

This project uses the extended implementation of the paper 
[Face Alignment in Full Pose Range: A 3D Total Solution
](https://arxiv.org/abs/1804.01005) located in the repository
[here](https://github.com/cleardusk/3DDFA_V2) and the previous 
implementation can be found [here](https://github.com/cleardusk/3DDFA).

# How to use the code

- **Step 1: Getting 3DDFA_V2 library**

    For getting 3DDFA_V2 library, use the following command in your python environment. 
    Refer the file original_readme.md or follow the steps on the 3DDFA_V2 github page 
    [here](https://github.com/cleardusk/3DDFA_V2).


- **Step 2: Using the main file**
    For generating points using the main file, edit the values of the
    following variables

    ```python
  # path to the input video
  in_vid_path = 'path/of/video/to/use.mp4'
  
  # keep this empty if you don't want to generate visualization
  out_vid_path = 'path/of/video/to/generate.mp4'
    
  # path to csv file for storing landmarks data
  out_csv_path = 'path/to/landmark/output.csv'
  
  # change the value of config file source
  config_file = 'model/config/to/use.yml'
  
  # type of landmark view set '2d_sparse' or '2d_dense'
  opt = 'ldmk_view_type'
    
  # flag for GPU usage set True or False
  use_GPU = 'gpu_flag'
  ```    

- **Step 3: Importing the main file**

    If you want to import the class and use it, create an object
    and pass the above values while calling the function generate_features().