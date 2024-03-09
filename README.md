# Segmentation of hyphae and yeast in fungi-infected tissue slice images and its application in analyzing anti-fungal blue light therapy

# Requirements
- Python 3.7
- tensorflow 2.7.0
- Some basic python packages, such as Numpy, OpenCV, SimpleITK.

# Training & Testing
- Train and test the models:  
    
`python train.py`  
  
The results will be save to `./model_results_dir`.  
  
- Evaluate the segmentation maps:  
    
You can evaluate the segmentation maps using the tool in `index_calculation.py`  
  
- Model code:  
  
The code for the compared models is in the other Python files
