# ---
# Title: "Titanic - Machine Learning from Disaster"
# Author: "Andrex Ibiza, MBA"
# Date: 2025-01-18
# Version: 2.1
# Score: "0.52870"
# Assessment: dramatically worse than v2.0
# for real don't use this notebook.
# editor_options: 
#   markdown: 
#     wrap: 72
# ---

# Load packages
library(caret)        # machine learning
library(dplyr)        # data manipulation
library(ggplot2)      # viz
library(Hmisc)        # robust describe() function
library(naniar)       # working with missing data
library(randomForest) # inference model

# Load train and test data
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
head(train) #--loaded successfully
head(test)  #--loaded successfully

# Evaluate structure and data types
# str(train)
# str(test)
# 
# describe(train)
# train has missing values: Age 177, Cabin 687, Embarked 2
# describe(test)
# test has missing values: Cabin 327, Fare 1, Age 86

# DATA CLEANING AND PREPROCESSING

# 1) Encode categorical variables
# [X] Encode Sex as numeric factor
train$Sex <- ifelse(train$Sex == "male", 1, 0)
test$Sex <- ifelse(test$Sex == "male", 1, 0)
# head(train[, "Sex"]) #--encoded successfully
# head(test[, "Sex"]) #--encoded successfully

# [X] Convert Pclass to an ordinal factor
train$Pclass <- factor(train$Pclass, levels = c(1, 2, 3), ordered = TRUE)
test$Pclass <- factor(test$Pclass, levels = c(1, 2, 3), ordered = TRUE)
# head(train[, "Pclass"]) #--encoded successfully
# head(test[, "Pclass"]) #--encoded successfully

# [X] One-hot encode Embarked
embarked_train_one_hot <- model.matrix(~ Embarked - 1, data = train)
embarked_test_one_hot <- model.matrix(~ Embarked - 1, data = test)

# Add the one-hot encoded columns back to the dataset
train <- cbind(train, embarked_train_one_hot)
test <- cbind(test, embarked_test_one_hot)

# Verify encoding:
head(train[, c("Embarked", "EmbarkedC", "EmbarkedQ", "EmbarkedS")])
head(test[, c("Embarked", "EmbarkedC", "EmbarkedQ", "EmbarkedS")])

# -- looks perfect, let's not forget about imputing our 2 missing values
# Impute 2 missing Embarked values with the mode
train$Embarked[train$Embarked == ""] <- NA
embarked_mode <- names(sort(table(train$Embarked)))
train$Embarked[is.na(train$Embarked)] <- embarked_mode

# verify imputation
describe(train$Embarked)

# now drop the original Embarked column
train <- train %>% select(-Embarked)
test <- test %>% select(-Embarked)
str(train)
str(test)

# 2) Apply log transformation to Fare
#--plot shape before transformation?
ggplot(train, aes(x = Fare)) +
  geom_histogram() +
  theme_minimal() +
  ggtitle("Fare (before transforming)")

#--note an extreme outlier over 500!
train$Fare <- log(train$Fare + 1)
test$Fare <- log(test$Fare + 1)
head(train[, "Fare"])
head(test[, "Fare"])

ggplot(train, aes(x = Fare)) +
  geom_histogram() +
  theme_minimal() +
  ggtitle("Log Transformed Fare")

# 3) Address missing values
# Age - Train
#--Predict missing ages using other features
train_age_data <- train %>% 
    select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)

# head(train[, c("Age", "Pclass", "Sex", "SibSp", "Parch", "Fare", "EmbarkedC", "EmbarkedQ", "EmbarkedS")])
#--verified that all these columns are formatted properly

train_age_complete <- train_age_data %>% filter(!is.na(Age))
train_age_missing <- train_age_data %>% filter(is.na(Age))

set.seed(666)
cv_control <- trainControl(method = "cv", number = 5)
train_age_cv_model <- train(
  Age ~ Pclass + Sex + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS,
  data = train_age_complete,
  method = "rf",
  trControl = cv_control,
  tuneLength = 3
)
print(train_age_cv_model)

# Use the best model to predict missing ages
predicted_train_ages <- predict(train_age_cv_model, newdata = train_age_missing)

# Impute the predicted ages back into the train dataset
train$Age[is.na(train$Age)] <- predicted_train_ages
describe(train$Age)

#--Age in test data
# Preprocess the test data for Age imputation
test_age_data <- test %>% 
  select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)

test_age_missing <- test_age_data %>% filter(is.na(Age))
test_age_complete <- test_age_data %>% filter(!is.na(Age))

# Use the trained train_age_cv_model to predict missing ages in the test dataset
predicted_test_ages <- predict(train_age_cv_model, newdata = test_age_missing)

# Impute the predicted ages back into the test dataset
test$Age[is.na(test$Age)] <- predicted_test_ages

n_miss(test$Age)

# Create HasCabin feature
# any_na(train$Cabin) # returns FALSE
# describe(train$Cabin) # 687 missing - need to replace empty string values

# Convert empty strings to NA in Cabin
train$Cabin[train$Cabin == ""] <- NA
test$Cabin[test$Cabin == ""] <- NA

n_miss(train$Cabin)
n_miss(test$Cabin)

# Encode the HasCabin variable:
train$HasCabin <- ifelse(!is.na(train$Cabin), 1, 0)
test$HasCabin <- ifelse(!is.na(test$Cabin), 1, 0)

# describe(train$HasCabin) # - perfect
head(train[, c("Cabin", "HasCabin")])  #looks good
head(test[, c("Cabin", "HasCabin")]) 

n_miss(train$HasCabin)
n_miss(test$HasCabin)

# Create the FamilySize feature
train$FamilySize <- train$SibSp + train$Parch + 1
test$FamilySize <- test$SibSp + test$Parch + 1

# Inspect the new feature
head(train[, "FamilySize"])
head(test[, "FamilySize"])

describe(train)
describe(test)
#--test still has 1 missing fare - impute with the median
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)
describe(test)

# Data preprocessing is now complete and we are ready to model 
# the `Survival` variable for the `test` dataset!

# MODEL SELECTION

#----------RANDOM FOREST-------best so far at R^2=0.4557431

# # Train the random forest model
# rf_cv_control <- trainControl(method = "cv", number = 5)
# set.seed(666)
# rf_model <- train(
#   Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize, 
#   data = train,
#   method = "rf",
#   trControl = rf_cv_control,
#   tuneLength = 5
# )
# 
# # Print the cross-validation results
# print(rf_model)
# 
# #---------XGBoost-------R^2 = 0.4523970
# # Train the XGBoost model
# library(xgboost)
# xgb_cv_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
# set.seed(666)
# xgb_model <- train(
#   Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize, 
#   data = train,
#   method = "xgbTree",  # Specify XGBoost
#   trControl = xgb_cv_control,
#   tuneLength = 5  # Explore a grid of hyperparameters
# )
# print(xgb_model)

#----------LOGISTIC REGRESSION-----R^2=0.4178629

# # Train the logistic regression model
# lr_cv_control <- trainControl(method = "cv", number = 5)
# set.seed(666)
# lr_model <- train(
#   Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize, 
#   data = train,
#   method = "glm",
#   family = "binomial",  # Specify logistic regression
#   trControl = lr_cv_control
# )
# print(lr_model)
# Generalized Linear Model 
# 
# 891 samples
# 11 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 713, 713, 713, 712, 713 
# Resampling results:
#   
#   RMSE       Rsquared   MAE      
# 0.3735736  0.4178629  0.2764035

#------Decision Tree-----R^2=0.4076239

# Train the decision tree model
# dt_cv_control <- trainControl(method = "cv", number = 5)
# set.seed(666)
# dt_model <- train(
#   Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize, 
#   data = train,
#   method = "rpart",  # Specify decision tree
#   trControl = dt_cv_control,
#   tuneLength = 5
# )
# 
# # Print the cross-validation results
# print(dt_model)

# CART 
# 
# 891 samples
# 11 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 713, 713, 713, 712, 713 
# Resampling results across tuning parameters:
#   
#   cp          RMSE       Rsquared   MAE      
# 0.01758003  0.3766414  0.4076239  0.2699124
# 0.02384903  0.3861017  0.3776516  0.2841644
# 0.03345148  0.3912430  0.3588676  0.2943822
# 0.07394186  0.3942795  0.3488828  0.3087028
# 0.29523072  0.4650045  0.2222900  0.4233506
# 
# RMSE was used to select the optimal model
# using the smallest value.
# The final value used for the model was cp
# = 0.01758003.
# > 

#----------LightGBM
# Convert categorical variables to factors (if not already done)
train$Survived <- as.numeric(train$Survived)  # Ensure Survived is numeric
train$Sex <- as.numeric(factor(train$Sex))    # Convert categorical columns
train$EmbarkedC <- as.numeric(factor(train$EmbarkedC))
train$EmbarkedQ <- as.numeric(factor(train$EmbarkedQ))
train$EmbarkedS <- as.numeric(factor(train$EmbarkedS))

# Create predictor matrix and target variable
X_train <- as.matrix(train[, c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
                               "EmbarkedC", "EmbarkedQ", "EmbarkedS", "HasCabin", "FamilySize")])
y_train <- train$Survived

library(lightgbm)

# Prepare LightGBM dataset
dtrain <- lgb.Dataset(data = X_train, label = y_train)
# 
# # Set parameters
# params <- list(
#   objective = "binary",
#   metric = "binary_error",  # Or "binary_logloss" for log loss
#   boosting = "gbdt",       # Gradient Boosting Decision Trees
#   learning_rate = 0.1,
#   num_leaves = 31,
#   max_depth = -1,
#   min_data_in_leaf = 20,
#   feature_fraction = 0.8
# )
# 
# # Perform cross-validation
# cv_results <- lgb.cv(
#   params = params,
#   data = dtrain,
#   nfold = 5,              # Number of folds
#   nrounds = 100,          # Number of boosting rounds
#   verbose = 1,
#   stratified = TRUE,      # Stratified sampling (useful for imbalanced data)
#   eval = "binary_error"   # Use binary error as the evaluation metric
# )

# Print cross-validation results
# print(cv_results)


#------HYPERPARAMETER TUNING


#------nfold
# Define the range of nfold values to test
# nfold_values <- 3:10
# results <- data.frame(nfold = integer(), best_score = numeric(), best_iter = integer())
# 
# # Loop through each nfold value
# for (n in nfold_values) {
#   cat("Testing nfold =", n, "\n")
#   
#   # Perform cross-validation with the current nfold
#   cv_results <- lgb.cv(
#     params = params,
#     data = dtrain,
#     nfold = n,
#     nrounds = 100,
#     verbose = 0,  # Suppress iteration-level output for clarity
#     stratified = TRUE,
#     eval = "binary_error"
#   )
#   
#   # Store results
#   results <- rbind(results, data.frame(
#     nfold = n,
#     best_score = cv_results$best_score,
#     best_iter = cv_results$best_iter
#   ))
# }
# 
# # Print the results
# print(results)
#---optimum nfold value is 8


#-----learning_rate
# Define learning rate values to test
# learning_rates <- c(0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
# 
# # Initialize results dataframe
# learning_rate_results <- data.frame(learning_rate = numeric(), best_score = numeric(), best_iter = numeric())
# 
# # Loop through each learning rate
# for (lr in learning_rates) {
#   cat("Testing learning_rate =", lr, "\n")
#   
#   # Update parameters with the current learning rate
#   params$learning_rate <- lr
#   
#   # Perform cross-validation
#   cv_results <- lgb.cv(
#     params = params,
#     data = dtrain,
#     nfold = 8,
#     nrounds = 100,
#     verbose = 0,  # Suppress iteration-level output
#     stratified = TRUE,
#     eval = "binary_error"
#   )
#   
#   # Record results
#   learning_rate_results <- rbind(learning_rate_results, data.frame(
#     learning_rate = lr,
#     best_score = cv_results$best_score,
#     best_iter = cv_results$best_iter
#   ))
# }
# 
# # Print results
# print(learning_rate_results)
# 
#   
# # Update learning rate in parameters
# params$learning_rate <- 0.03

# # Final cross-validation
# final_cv_results <- lgb.cv(
#   params = params,
#   data = dtrain,
#   nfold = 8,
#   nrounds = 200,  # Extend rounds to fully utilize lower learning rate
#   verbose = 1,
#   stratified = TRUE,
#   eval = "binary_error",
#   early_stopping_rounds = 10
# )
# 
# # Print the final cross-validation results
# print(final_cv_results)

#---ideal learning rate = 0.03

# Update learning rate in parameters
# Set parameters
# params <- list(
#   objective = "binary",
#   metric = "binary_error",  # Or "binary_logloss" for log loss
#   boosting = "gbdt",       # Gradient Boosting Decision Trees
#   learning_rate = 0.3,
#   num_leaves = 31,
#   max_depth = -1,
#   min_data_in_leaf = 20,
#   feature_fraction = 0.8
# )
# 
# # Final cross-validation
# final_cv_results <- lgb.cv(
#   params = params,
#   data = dtrain,
#   nfold = 8,
#   nrounds = 200,  # Extend rounds to fully utilize lower learning rate
#   verbose = 1,
#   stratified = TRUE,
#   eval = "binary_error",
#   early_stopping_rounds = 10
# )

# Print the final cross-validation results
# print(final_cv_results)
#----best_score: 0.170591289341289

#-------num_leaves
# 
# # Define a range of num_leaves values to test
# num_leaves_values <- c(15, 31, 63, 127, 255)
# 
# # Initialize results dataframe
# num_leaves_results <- data.frame(num_leaves = integer(), best_score = numeric(), best_iter = integer())
# 
# # Loop through each num_leaves value
# for (leaves in num_leaves_values) {
#   cat("Testing num_leaves =", leaves, "\n")
#   
#   # Update parameters with the current num_leaves
#   params$num_leaves <- leaves
#   
#   # Perform cross-validation
#   cv_results <- lgb.cv(
#     params = params,
#     data = dtrain,
#     nfold = 8,
#     nrounds = 200,
#     verbose = 0,  # Suppress iteration-level output
#     stratified = TRUE,
#     eval = "binary_error",
#     early_stopping_rounds = 10
#   )
#   
#   # Record results
#   num_leaves_results <- rbind(num_leaves_results, data.frame(
#     num_leaves = leaves,
#     best_score = cv_results$best_score,
#     best_iter = cv_results$best_iter
#   ))
# }
# 
# # Print results
# print(num_leaves_results)
#----31 is already the optimum number

# num_leaves best_score best_iter
# 1         15  0.1651358         6
# 2         31  0.1605031        15
# 3         63  0.1705202        11
# 4        127  0.1728333        12
# 5        255  0.1717845        13

# #--------max_depth
# # Define a range of max_depth values to test
# max_depth_values <- c(-1, 3, 5, 7, 10, 15, 20)
# 
# # Initialize results dataframe
# max_depth_results <- data.frame(max_depth = integer(), best_score = numeric(), best_iter = integer())
# 
# # Loop through each max_depth value
# for (depth in max_depth_values) {
#   cat("Testing max_depth =", depth, "\n")
#   
#   # Update parameters with the current max_depth
#   params$max_depth <- depth
#   
#   # Perform cross-validation
#   cv_results <- lgb.cv(
#     params = params,
#     data = dtrain,
#     nfold = 8,
#     nrounds = 200,
#     verbose = 0,  # Suppress iteration-level output
#     stratified = TRUE,
#     eval = "binary_error",
#     early_stopping_rounds = 10
#   )
#   
#   # Record results
#   max_depth_results <- rbind(max_depth_results, data.frame(
#     max_depth = depth,
#     best_score = cv_results$best_score,
#     best_iter = cv_results$best_iter
#   ))
# }
# 
# # Print results
# print(max_depth_results)
# --- max depth = 20

#-----final tuned run
# Set parameters
# params <- list(
#   objective = "binary",
#   metric = "binary_error",  # Or "binary_logloss" for log loss
#   boosting = "gbdt",       # Gradient Boosting Decision Trees
#   learning_rate = 0.3,
#   num_leaves = 31,
#   max_depth = 20,
#   min_data_in_leaf = 20,
#   feature_fraction = 0.8
# )
# 
# # Final cross-validation
# final_cv_results <- lgb.cv(
#   params = params,
#   data = dtrain,
#   nfold = 8,
#   nrounds = 200,  # Extend rounds to fully utilize lower learning rate
#   verbose = 1,
#   stratified = TRUE,
#   eval = "binary_error",
#   early_stopping_rounds = 10
# )
# 
# # Print the final cross-validation results
# print(final_cv_results)

#-------Bayesian optimization
# library(mlrMBO)
# library(mlr)
# library(ParamHelpers)
# 
# # Define the parameter search space
# param_set <- makeParamSet(
#   makeNumericParam("learning_rate", lower = 0.01, upper = 0.3),
#   makeIntegerParam("num_leaves", lower = 15, upper = 63),
#   makeNumericParam("feature_fraction", lower = 0.6, upper = 0.9)
# )
# 
# # Define the objective function
# objective_function <- makeSingleObjectiveFunction(
#   fn = function(x) {
#     # Set parameters for LightGBM
#     params <- list(
#       objective = "binary",
#       metric = "binary_error",
#       boosting = "gbdt",
#       learning_rate = x["learning_rate"],
#       num_leaves = as.integer(x["num_leaves"]),
#       feature_fraction = x["feature_fraction"]
#     )
#     # Perform cross-validation
#     cv_results <- lgb.cv(
#       params = params,
#       data = dtrain,
#       nfold = 8,
#       nrounds = 200,
#       verbose = 0,
#       stratified = TRUE,
#       eval = "binary_error",
#       early_stopping_rounds = 10
#     )
#     return(cv_results$best_score)
#   },
#   par.set = param_set,
#   minimize = TRUE
# )
# 
# # Set up the control parameters
# control <- makeMBOControl()
# control <- setMBOControlTermination(control, iters = 20)  # Number of iterations
# control <- setMBOControlInfill(control, crit = makeMBOInfillCritEI())  # Expected improvement criterion
# 
# # Perform the optimization
# result <- mbo(fun = objective_function, control = control, design = generateDesign(10, param_set))
# print(result)
# 
# # Recommended parameters:
# #   learning_rate=0.0226; num_leaves=61; feature_fraction=0.809
# # Objective: y = 0.162
# 
# # Recommended parameters
final_params <- list(
  objective = "binary",
  metric = "binary_error",  # Binary error as the evaluation metric
  boosting = "gbdt",       # Gradient Boosting Decision Trees
  learning_rate = 0.0226,   # Recommended learning rate
  num_leaves = 61,          # Recommended number of leaves
  feature_fraction = 0.809  # Recommended feature fraction
)

# Train the final model with cross-validation
ultimate_model_cv <- lgb.cv(
  params = final_params,
  data = dtrain,
  nfold = 8,
  nrounds = 2000,           # Allow enough boosting rounds for convergence
  verbose = 1,              # Display iteration logs
  stratified = TRUE,
  eval = "binary_error",
  early_stopping_rounds = 50 # Stop early if no improvement is seen
)

# Print the best score and iteration
cat("Best Binary Error:", ultimate_model_cv$best_score, "\n")
cat("Best Iteration:", ultimate_model_cv$best_iter, "\n")


#------------PREDICT!

# Train the final LightGBM model using the best number of iterations
final_model <- lgb.train(
  params = final_params,
  data = dtrain,
  nrounds = ultimate_model_cv$best_iter,  # Use the best number of iterations
  verbose = 1
)

# Prepare test data (similar to training data)
X_test <- as.matrix(test[, c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
                             "EmbarkedC", "EmbarkedQ", "EmbarkedS", "HasCabin", "FamilySize")])

# Predict probabilities
predictions <- predict(final_model, X_test)

# Convert probabilities to binary outcomes (0 or 1)
binary_predictions <- ifelse(predictions >= 0.5, 1, 0)

# Add predictions to the test dataset
test$Survived <- binary_predictions
table(test$Survived)

# Save the updated test dataset with predictions
gender_submission <- test %>% select(PassengerId, Survived)
table(gender_submission)
write.csv(gender_submission, "submission.csv", row.names = FALSE)


