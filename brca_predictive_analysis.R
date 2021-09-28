# Breast Cancer Diagnosis Biopsy Prediction

# Importing Libraries
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)

# Importing Dataset
data(brca)

#-------------------------------------------------------------------------------

# FEATURE ENGINEERING

# Scaling the matrix
x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

# Calculating the distance between the samples

# Benign Samples
dist_BtoB <- as.matrix(d)[1, brca$y == "B"]
mean(dist_BtoB[2:length(dist_BtoB)])

# Malignant Samples
dist_BtoM <- as.matrix(d)[1, brca$y == "M"]
mean(dist_BtoM)


# Relationship between features using the scaled matrix using a heatmap
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features))

# Performing hierarchical clustering on the 30 features
h <- hclust(d_features)
plot(h)

# Performing principal component analysis of the scaled matrix
pc <- prcomp(x_scaled)
summary(pc)$importance

# Plotting the first 2 PCs
data.frame(pc_1 = pc$x[,1], pc_2 = pc$x[,2], 
           type = brca$y) %>%
            ggplot(aes(pc_1, pc_2, color = type)) +
            geom_point()

# Box plot of all 30 PCs
data.frame(type = brca$y, pc$x[,1:30]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()

#-------------------------------------------------------------------------------

# Splitting Data into Training and Test Set
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

train_set <- data.frame(train_x, train_y)
test_set <- data.frame(test_x, test_y)

#-------------------------------------------------------------------------------

# MODEL PREDICTIONS

# K-means Clustering Model
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}

set.seed(3, sample.kind = "Rounding")    # if using R 3.6 or later
k <- kmeans(train_x, centers = 2)
y_kmeans <- predict_kmeans(test_x,k)
y_kmeans <-ifelse(y_kmeans == 1, "B", "M") %>% factor(levels = levels(test_y))

# Accuracy
mean(y_kmeans == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_kmeans), test_y)
specificity(factor(y_kmeans), test_y)

#-------------------------------------------------------------------------------

# Logistic Regression Model

# Using all predictors
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
train_glm <- train(train_y ~ ., method = "glm", data = train_set)

# Obtain predictors and accuracy
y_glm <- predict(train_glm, test_set) %>% factor(levels = levels(test_y))

# Accuracy
mean(y_glm == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_glm), test_y)
specificity(factor(y_glm), test_y)

# Variable Importance for this model
imp <- varImp(train_glm)
imp

#-------------------------------------------------------------------------------

# LDA and QDA Models

# LDA
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
train_lda <- train(train_x, train_y, method = "lda")

# Obtain predictors and accuracy
y_lda <- predict(train_lda, test_x) %>% factor(levels = levels(test_y))

# Accuracy
mean(y_lda == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_lda), test_y)
specificity(factor(y_lda), test_y)

# Variable Importance for this model
imp <- varImp(train_lda)
imp

#----------------------------------------------------------#

# QDA
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
train_qda <- train(train_x, train_y, method = "qda")

# Obtain predictors and accuracy
y_qda <- predict(train_qda, test_x) %>% factor(levels = levels(test_y))

# Accuracy
mean(y_qda == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_qda), test_y)
specificity(factor(y_qda), test_y)

# Variable Importance for this model
imp <- varImp(train_qda)
imp

#-------------------------------------------------------------------------------

# Loess model

set.seed(5, sample.kind="Rounding") # if using R 3.6 or later
train_loess <- train(train_x, train_y, method = "gamLoess")

# Obtain predictors and accuracy
y_loess <- predict(train_loess, test_x) %>% factor(levels = levels(test_y))

# Accuracy
mean(y_loess == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_loess), test_y)
specificity(factor(y_loess), test_y)

#-------------------------------------------------------------------------------

# K-Nearest Neighbors Model

set.seed(7, sample.kind="Rounding") # if using R 3.6 or later
train_knn <- train(train_x, train_y, method = "knn", tuneGrid = data.frame(k = seq(3, 21, 2)))

# Find the optimal K value
train_knn$bestTune

ks <- train_knn$results$k
accuracy <- train_knn$results$Accuracy
qplot(ks, accuracy, geom = 'line')

# Obtain predictors and accuracy
y_knn <- predict(train_knn, test_x) %>% factor(levels = levels(test_y))

# Accuracy
mean(y_knn == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_knn), test_y)
specificity(factor(y_knn), test_y)

# Variable Importance for this model
imp <- varImp(train_knn)
imp

#-------------------------------------------------------------------------------

# Random forest model

set.seed(9, sample.kind="Rounding") # if using R 3.6 or later
train_rf <- train(train_x, train_y, method = "rf", tuneGrid = data.frame(mtry = c(3, 5, 7, 9)), importance = TRUE)

# Find the optimal mtry value
train_rf$bestTune

mtry <- train_rf$results$mtry
accuracy <- train_rf$results$Accuracy
qplot(mtry, accuracy, geom = 'line')

# Obtain predictors and accuracy
y_rf <- predict(train_rf, test_x) %>% factor(levels = levels(test_y))

# Accuracy
mean(y_rf == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_rf), test_y)
specificity(factor(y_rf), test_y)

# Variable Importance for this model
imp <- varImp(train_rf)
imp

#-------------------------------------------------------------------------------

# Ensemble
y_pred <- data.frame(y_kmeans, y_glm, y_lda, y_qda, y_loess, y_knn, y_rf)
types <- rowMeans(y_pred == "M")
y_ess <- ifelse(types > 0.5, "M", "B") %>% factor(levels = levels(test_y))

# Accuracy
mean(y_ess == test_y)

# Specificity and Sensitivity
sensitivity(factor(y_ess), test_y)
specificity(factor(y_ess), test_y)

#-------------------------------------------------------------------------------

# Compiling the results of all the Models
models <- c("K means", "Logistic regression", "LDA", "QDA", "Loess", 
            "K nearest neighbors", "Random forest", "Ensemble")
accuracy <- c(mean(y_kmeans == test_y),
              mean(y_glm == test_y),
              mean(y_lda == test_y),
              mean(y_qda == test_y),
              mean(y_loess == test_y),
              mean(y_knn == test_y),
              mean(y_rf == test_y),
              mean(y_ess == test_y))

spec <- c(specificity(factor(y_kmeans), test_y),
          specificity(factor(y_glm), test_y),
          specificity(factor(y_lda), test_y),
          specificity(factor(y_qda), test_y),
          specificity(factor(y_loess), test_y),
          specificity(factor(y_knn), test_y),
          specificity(factor(y_rf), test_y),
          specificity(factor(y_ess), test_y))

sens <- c(sensitivity(factor(y_kmeans), test_y),
          sensitivity(factor(y_glm), test_y),
          sensitivity(factor(y_lda), test_y),
          sensitivity(factor(y_qda), test_y),
          sensitivity(factor(y_loess), test_y),
          sensitivity(factor(y_knn), test_y),
          sensitivity(factor(y_rf), test_y),
          sensitivity(factor(y_ess), test_y))

data.frame(Model = models, Accuracy = accuracy, Specificity = spec, Sensitivity = sens)
