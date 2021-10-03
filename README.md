# Breast Cancer Diagnosis Biopsy Prediction

Machine Learning: Predicting Breast Cancer Diagnosis Using Multiple Techniques<br />
Tools: R (Dslabs, Caret, Tidyverse, MatrixStats, Gam)<br />
Data Source: Dslabs<br />
<br />

**CONTENT**
- Exploratory Data Analysis and Feature Selection
- Machine Learning Model Building / Training
-	Prediction / Accuracy

**EXPLORATORY DATA ANALYSIS AND FEATURE SELECTION**<br />
Before training and testing a machine learning model, it is important to understand the data to be used. This is the purpose of exploratory data analysis. 
The training dataset consist of 50 explanatory variables and 1 prediction variable describing if the tumors are benign (not cancer) or malignant (cancer). 
-	brca$y: a vector of sample classifications ("B" = benign or "M" = malignant)
-	brca$x: a matrix of numeric features describing properties of the shape and size of cell nuclei extracted from biopsy microscope image<br />

The first step to this analysis was to standardize and scale the matrix of numerical features and to calculate the distance between the tumor samples. 
This was done through the following steps;

![](image/scale.png)
