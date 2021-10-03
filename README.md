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

By careful examination and preprocessing, relevant features were selected and used to train a model to predict the diagnosis of each tumor samples. In other to select the relevant features for prediction, it is important to identity features that do not have a strong correlation with each other. For the numerical variables, this was done by using a correlation heatmap

![](image/rlcode.png)

![](image/heatmap.jpeg)

Visualizing the correlation between the numerical features using the correlation heatmap above, I decided to select 30 variables as they were the variables that were least correlated with each other. Furthermore, I also performed hierarchical clustering on the 30 features to determine their relationship.

![](image/hclcode.png)

![](image/hclust_plot.jpeg)

Principal component analysis was performed on the scaled matrix showing the standard deviation, proportion of variance, and the cumulative proportion. The first and last two principal components were plotted in a scattered plot as well as a boxplot to show the relationship between themselves and the tumor type.
