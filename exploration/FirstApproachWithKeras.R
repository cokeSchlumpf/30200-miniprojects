#--------------------------------
# Keras Neural Net Approach     #
#--------------------------------

#--------------------------------
# Load Libraries:
library(caret)
library(keras)
library(dplyr)

rm(list = ls())

#--------------------------------
#Load Datasets:
setwd("~/Studium/DataScience/HS Alb. Sig/Machine Learning 30200/Präsenzwochenende/Data Mining Mini-Projekt/data_miniproject_stud/data_miniproject_stud/comp1")

df_raw_risse_train <- read.csv("data_risse.csv", header=F, sep = ";")
df_raw_risse_test <- read.csv("data_risse_test.csv", header=F, sep = ";")
str(df_raw_risse_train)
df_raw_risse_train[1:5,c(1:10)]


#--------------------------------
# Data Prepartion incl. data split

df_raw_risse_train$ID <- seq.int(nrow(df_raw_risse_train))

set.seed(42)
index <- createDataPartition(df_raw_risse_train$ID, p = 0.8, list = FALSE)
x_fit_fx_vec <- df_raw_risse_train[index, -c(1)]
x_validate_fx_vec  <- df_raw_risse_train[-index, -c(1)]
# y_fit_response <- df_raw_risse_train[index,1]
# y_validate_response <- df_raw_risse_train[-index,1]

y_fit_response <- to_categorical(df_raw_risse_train[index,1])
y_validate_response <- to_categorical(df_raw_risse_train[-index,1])
# works for data frame not for matrix:
# names(y_validate_response)[1] <- "KeinRiss"
# names(y_validate_response)[2] <- "Riss"
# Therefore:
colnames(y_validate_response) <- c("actual_kein_riss_cat","actual_riss_cat")
colnames(y_fit_response) <- c("actual_kein_riss_cat","actual_riss_cat")

#Checks:
max(x_fit_fx_vec)
max(x_validate_fx_vec)
dim(x_fit_fx_vec)
dim(x_validate_fx_vec);
dim(y_fit_response)
dim(y_validate_response);



#--------------------------------
# Design the Neural Net
model <- keras_model_sequential() 
model %>% 
#   layer_dense(units = 256, activation = 'relu', input_shape = c(513)) %>% 
#   layer_dropout(rate = 0.4) %>% 
#   layer_dense(units = 128, activation = 'relu') %>%
#   layer_dropout(rate = 0.3) %>%
#   layer_dense(units = 2, activation = 'softmax')
  layer_dense(units = 256, activation = 'relu', input_shape = c(513)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)
#--------------------------------
# Compile the Neural Net
model %>% compile(
  loss = 'categorical_crossentropy', #https://keras.rstudio.com/reference/loss_mean_squared_error.html
  optimizer = optimizer_rmsprop(), #binary__crossentropy #Adam ist aktuell i.d.R der beste
  metrics = c('accuracy')
)

#--------------------------------
# Training and Evaluation
x_fit_fx_vec_mat <- data.matrix(x_fit_fx_vec)
y_fit_response_mat <- data.matrix(y_fit_response)

history <- model %>% fit(
  x_fit_fx_vec_mat, y_fit_response_mat, 
  epochs = 10,
  batch_size = 128, 
  validation_split = 0.2
)

plot(history)

#--------------------------------
# Generate predictions on validation dataset:
x_validate_fx_vec_mat <- data.matrix(x_validate_fx_vec)
#predict_on_vds <- model %>% predict_classes(x_validate_fx_vec_mat)
predict_on_vds <- model %>% predict(x_validate_fx_vec_mat)

#--------------------------------
# Compare Actual vs. prediction:
vds_predict_vs_actual <- cbind(predict_on_vds, y_validate_response)
df_vds_predict_vs_actual <- as.data.frame(as.table(vds_predict_vs_actual))

df_vds_predict_vs_actual <- as.data.frame(vds_predict_vs_actual)

df_vds_predict_vs_actual <- df_vds_predict_vs_actual %>%
  rename(
    predicted_kein_riss_con = V1,
    predicted_riss_con = V2
  ) %>% 
  mutate(
    predicted_riss_cat = if_else(predicted_riss_con > 0.5, 1, 0),
    residuen_con = abs(predicted_riss_con - actual_riss_cat),
    residuen_cat = abs(predicted_riss_cat - actual_riss_cat)
  )

View(df_vds_predict_vs_actual)
sum(df_vds_predict_vs_actual$residuen_cat)/nrow(df_vds_predict_vs_actual)

#--------------------------------
# Model Performance Analysis
model %>% evaluate(x_fit_fx_vec_mat, y_fit_response_mat)
