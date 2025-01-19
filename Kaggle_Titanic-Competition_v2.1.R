# ---
# Title: "Titanic - Machine Learning from Disaster"
# Author: "Andrex Ibiza, MBA"
# Date: 2025-01-18
# Version: 2
# Score: "0.76076"
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

#--debugged up to this point so far-------

describe(train)
describe(test)
#--test still has 1 missing fare - impute with the median
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)
describe(test)

#---------removing these features below is a mistake.
# --------rerun above this line
# instead, specifically select features within the model instead
# of using the Survival ~ . syntax

# 5) Remove unnecessary features
# train_trimmed <- train %>% 
#   select(-Cabin) %>% 
#   select(-Name) %>% 
#   select(-PassengerId) %>% 
#   select(-Ticket)
# 
# test_trimmed <- test %>% 
#   select(-Cabin) %>% 
#   select(-Name) %>% 
#   select(-PassengerId) %>% 
#   select(-Ticket)
# 
# str(test)
# str(train)

# Data preprocessing is now complete and we are ready to model 
# the `Survival` variable for the `test` dataset!

# Train the random forest model
rf_cv_control <- trainControl(method = "cv", number = 5)
set.seed(666)
rf_model <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize, 
  data = train,
  method = "rf",
  trControl = rf_cv_control,
  tuneLength = 5
)

# Print the cross-validation results
print(rf_model)

#---tuneLength = 3
# Random Forest 
# 
# 891 samples
# 11 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 713, 713, 713, 713, 712 
# Resampling results across tuning parameters:
#   
#   mtry  RMSE       Rsquared   MAE      
# 2    0.3663029  0.4506800  0.2965819
# 7    0.3649883  0.4482027  0.2465107
# 12    0.3699093  0.4386277  0.2435517
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 7.

#------tuneLength = 5
# Random Forest 
# 
# 891 samples
# 11 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 713, 713, 713, 712, 713 
# Resampling results across tuning parameters:
#   
#   mtry  RMSE       Rsquared   MAE      
# 2    0.3647472  0.4557431  0.2949507
# 4    0.3612627  0.4532478  0.2558578
# 7    0.3653994  0.4443990  0.2456134
# 9    0.3678295  0.4395726  0.2434292
# 12    0.3706840  0.4323223  0.2431312
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 4.

# Use the trained model to predict Survived in the test dataset
test$Survived <- predict(rf_model, newdata = test)

describe(test$Survived)

# Round values in test$Survived to 0 or 1
test$Survived <- ifelse(test$Survived >= 0.5, 1, 0)

# Check the updated values
table(test$Survived)

# Save the updated test dataset with predictions
gender_submission <- test %>% select(PassengerId, Survived)
head(gender_submission)
write.csv(gender_submission, "gender_submission.csv", row.names = FALSE)


