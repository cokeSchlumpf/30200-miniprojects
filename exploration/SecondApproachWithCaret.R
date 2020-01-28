rm(list = ls())

#--------------------------------
# Load Libraries:
library(caret)
library(dplyr)
library(ggplot2)
library(tictoc)
library(MLmetrics)


#--------------------------------
#Load Datasets:

#-----------
#Load: Comp. 1 Data - Risse
setwd("~/Studium/DataScience/HS Alb. Sig/Machine Learning 30200/Präsenzwochenende/Data Mining Mini-Projekt/data_miniproject_stud/data_miniproject_stud/comp1")
df_raw_risse_train <- read.csv("data_risse.csv", header=F, sep = ";")
df_raw_risse_test <- read.csv("data_risse_test.csv", header=F, sep = ";")
df_raw_feature_vec_train <-df_raw_risse_train
df_raw_feature_vec_test <-df_raw_risse_test
selected_competition <- "comp1_risse"

#-----------
#Load: Comp. 2 Data - Schalungsanker
setwd("~/Studium/DataScience/HS Alb. Sig/Machine Learning 30200/Präsenzwochenende/Data Mining Mini-Projekt/data_miniproject_stud/data_miniproject_stud/comp2")
df_raw_anker_train <- read.csv("data_schalungsanker.csv", header=F, sep = ";")
df_raw_anker_test <- read.csv("data_schalungsanker_test.csv", header=F, sep = ";")
df_raw_feature_vec_train <-df_raw_anker_train
df_raw_feature_vec_test <-df_raw_anker_test
selected_competition <- "comp2_anker"

#-----------

# str(df_raw_feature_vec_train)
# df_raw_feature_vec_train[1:5,c(1:10)]
summary_df_raw_risse_train <- summary(df_raw_feature_vec_train)
#View(summary_df_raw_risse_train)

names(df_raw_feature_vec_train) <- c("actual_cat", names(df_raw_feature_vec_test))
names(df_raw_feature_vec_train)
names(df_raw_feature_vec_test)
df_raw_feature_vec_train$ID <- seq.int(nrow(df_raw_feature_vec_train))


df_feature_vec_train <- df_raw_feature_vec_train %>%
  mutate(
    actual_cat = as.factor(actual_cat)
  )

df_feature_vec_test <- df_raw_feature_vec_test

#Remove Feature Vectors with only NA or only 0, these columns have no information value
# columns_with_no_value <- sapply(df_feature_vec_train, function(x)all(is.na(x) | x == 0))
# df_feature_vec_train <- Filter(function(x) !(all(x==""|x==0)), df_feature_vec_train)
# sum(columns_with_no_value)
# ncol(df_feature_vec_train)
# 
# df_feature_vec_test <- Filter(function(x) !(all(x==""|x==0)), df_feature_vec_test)
# df_feature_vec_test
# ncol(df_feature_vec_test)


#--------------------------------
# Data Prepartion incl. data split

df_feature_vec_train$ID <- seq.int(nrow(df_feature_vec_train))

set.seed(42)
index <- createDataPartition(df_feature_vec_train$ID, p = 0.90, list = FALSE)
yx_fit <- df_feature_vec_train[index,]
yx_validate  <- df_feature_vec_train[-index,]

yx_fit <- yx_fit[,-which(names(yx_fit)=="ID")]
yx_validate <- yx_validate[-which(names(yx_validate)=="ID")]

#Checks:
max(yx_fit[,-which(names(yx_fit)=="actual_cat")])
max(yx_validate[,-which(names(yx_fit)=="actual_cat")])
dim(yx_fit)
dim(yx_validate)
dim(df_feature_vec_test)
anyNA(yx_fit)
anyNA(yx_validate)

#--------------------------------
# Train Model

# Simple GBM Train
train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)

set.seed(123)
tic("Runtime - Train Simple GBM")
fitted_s_gbm <- train(actual_cat ~ .,
                      data = yx_fit, 
                      method = "gbm", 
                      trControl = train_ctrl,
                      verbose = FALSE)
fitted_model <- fitted_s_gbm
fitted_s_gbm
toc()

#--------
# More advanced GBM Train

train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)


gbmGrid <-  expand.grid(interaction.depth = c(10),#,12,14), 
                        n.trees = (2:4)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

set.seed(234)
tic("Runtime - Train more advanced GBM")
fitted_a_gbm <- train(actual_cat ~ .,
                      data = yx_fit, 
                      method = "gbm", 
                      trControl = train_ctrl, 
                      verbose = FALSE, 
                      ## Now specify the exact models 
                      ## to evaluate:
                      tuneGrid = gbmGrid)
fitted_model <- fitted_a_gbm
fitted_a_gbm
toc()

plot(fitted_a_gbm)


#--------
# Extrem Gradient Boosting

train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)

xgbGrid <-  expand.grid(
  nrounds = c(10),#,20,30,50, 100), # (# Boosting Iterations)
  max_depth = 4:8, # (Max Tree Depth)
  eta = c(0.075, 0.1) , #(Shrinkage)
  gamma  = 0 , #(Minimum Loss Reduction)
  colsample_bytree = c(0.3, 0.4, 0.5), #(Subsample Ratio of Columns)
  min_child_weight = c(2.0, 2.25), #(Minimum Sum of Instance Weight)
  subsample = 1 #(Subsample Percentage)
)
dim(xgbGrid)

set.seed(345)
tic("Runtime - Extrem Gradient Boosting")
fitted_xgbTree <- train(actual_cat ~ .,
                        data = yx_fit, 
                        method = "xgbTree",
                        tuneGrid =xgbGrid,
                        trControl = train_ctrl,
                        verbose = FALSE)
fitted_model <- fitted_xgbTree
fitted_xgbTree
plot(fitted_xgbTree)
toc()

#--------
#Random Forest
train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)

set.seed(456)
tic("Runtime - Train Random Forest")
fitted_rf <- train(actual_cat ~ .,
                   data = yx_fit, 
                   method = "rf", 
                   trControl = train_ctrl,
                   verbose = FALSE)
fitted_model <- fitted_rf
fitted_rf
toc()

#--------
#Random Forest (Ranger Implementation)
train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)

set.seed(456)
tic("Runtime - Train Ranger Random Forest Version")
fitted_rf_ranger <- train(actual_cat ~ .,
                   data = yx_fit, 
                   method = "ranger", 
                   trControl = train_ctrl,
                   verbose = FALSE)
fitted_model <- fitted_rf_ranger
fitted_rf
toc()

#--------
# SVM with linear Kernel

train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)

set.seed(456)
tic("Runtime - Train SVM with linear Kernel")
fitted_svmLinear <- train(actual_cat ~ .,
                          data = yx_fit, 
                          method = "svmLinear3", 
                          trControl = train_ctrl,
                          verbose = FALSE)
fitted_model <- fitted_svmLinear
fitted_rf
toc()

#--------
#SVM with Least Squares with Polynomial Kernel

train_ctrl <- trainControl(## 10-fold CV
  method = "repeatedcv",#"repeatedcv",
  number = 5,
  repeats = 2
)

set.seed(456)
tic("Runtime - Train SVM with polynominal Kernel")
fitted_svm_poly <- train(actual_cat ~ .,
                          data = yx_fit, 
                          method = "lssvmPoly", 
                          trControl = train_ctrl,
                          verbose = FALSE)
fitted_model <- fitted_svm_poly
fitted_rf
toc()

#--------------------------------
# Generate predictions on validation dataset:

predict_on_vds <- yx_validate %>%
  select(
    actual_cat
  ) %>% 
  mutate(
    predicted_cat = predict(fitted_model, newdata = yx_validate[,-which(names(yx_validate)=="actual_cat")]),
  )

#--------------------------------
# Compare Actual vs. prediction:

predict_on_vds <- predict_on_vds %>%
  mutate(
    #predicted_cat = if_else(predicted_con > 0.5, 1, 0),
    #residuen_con = abs(predicted_con - actual_cat),
    residuen_cat = as.factor(if_else(predicted_cat != actual_cat, 1, 0)),
    residuen_cat_label = as.factor(if_else(residuen_cat == 0, "Correct","Wrong")),
    actual_cat_label = if_else(actual_cat == 1, "positiv","negativ"),
    predicted_cat_label = as.factor(if_else(actual_cat == 1, "positiv","negativ"))
  )


#View(predict_on_vds)
sum(predict_on_vds$residuen_cat)/nrow(predict_on_vds)

# ggplot(predict_on_vds, aes(residuen_con)) +
#   geom_histogram(bins = 500)
# 
# 
# ggplot(predict_on_vds, aes(predicted_con, actual_cat)) +
#   geom_point()

caret::confusionMatrix(as.factor(predict_on_vds$predicted_cat), as.factor(predict_on_vds$actual_cat))

t1_value_on_traindata <- F1_Score(y_pred = predict_on_vds$predicted_cat, y_true = predict_on_vds$actual_cat) #, positive = "1")
t1_value_on_traindata
#--------------------------------
# Predict on test dataset

predict_on_tds <- df_feature_vec_test %>%
  mutate(
    predicted_cat = predict(fitted_model, newdata = df_feature_vec_test)
  ) %>%
  mutate(
    predicted_cat = as.integer(as.character(predicted_cat))
  ) %>%
  select(
    predicted_cat
  )

filename <- paste(selected_competition, "_estimated_t1-value_", t1_value_on_traindata, ".csv", sep = "")
write.csv(predict_on_tds, paste('C:/Users/Staab/Documents/Studium/DataScience/HS Alb. Sig/Machine Learning 30200/Präsenzwochenende/Data Mining Mini-Projekt/Development/',filename, sep =""), row.names = FALSE, col.names = FALSE, sep = ";")
