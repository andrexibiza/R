---
Title: "Titanic - Machine Learning from Disaster"
Author: "Andrex Ibiza, MBA"
Date: 2025-01-14
Version: 2
Score: "TBD"
editor_options: 
  markdown: 
    wrap: 72
---

# Load packages
library(caret)        # machine learning
library(dplyr)        # data manipulation
library(ggplot2)      # viz
library(Hmisc)        # robust describe() function
library(naniar)       # working with missing data
library(randomForest) # inference model
library(tidyr)        # data cleaning

# Load `train.csv` data
train <- read.csv("train.csv", stringsAsFactors = FALSE)
head(train, 5) #--loaded successfully

# Evaluate structure and data types
str(train)
describe(train)

# DATA CLEANING AND PREPROCESSING

# 1) Encode categorical variables
# [X] Encode Sex as numeric factor
train$Sex <- ifelse(train$Sex == "male", 1, 0)
# head(train) #--encoded successfully

# [X] Convert Pclass to an ordinal factor
train$Pclass <- factor(train$Pclass, levels = c(1, 2, 3), ordered = TRUE)
# head(train) #--encoded successfully

# [X] One-hot encode Embarked
embarked_one_hot <- model.matrix(~ Embarked - 1, data = train)

# Add the one-hot encoded columns back to the dataset
train <- cbind(train, embarked_one_hot)

# Verify encoding:
# head(train[, c("Embarked", "EmbarkedC", "EmbarkedQ", "EmbarkedS")]) 
# -- looks perfect, now drop the original Embarked column
train <- train %>% select(-Embarked)
# head(train)

# 2) Apply log transformation to Fare
train$Fare <- log(train$Fare + 1)
head(train[, "Fare"])
# -- is the result normally distributed?
ggplot(train, aes(x = Fare)) +
  geom_histogram()
# -- the answer is a hard no. the data are still very right-skewed with outliers.

# 3) Address missing values
# Age
#--Predict missing ages using other features
age_data <- train %>% 
    select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)

age_complete <- age_data %>% filter(!is.na(Age))
age_missing <- age_data %>% filter(is.na(Age))

set.seed(666)
cv_control <- trainControl(method = "cv", number = 5)
age_cv_model <- train(
  Age ~ Pclass + Sex + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS,
  data = age_complete,
  method = "rf",
  trControl = cv_control,
  tuneLength = 3
)
print(age_cv_model)

# Use the best model to predict missing ages
predicted_ages <- predict(age_cv_model, newdata = age_missing)

# Impute the predicted ages back into the train dataset
train$Age[is.na(train$Age)] <- predicted_ages
describe(train$Age)

# Create HasCabin feature
# any_na(train$Cabin) # returns FALSE
# describe(train$Cabin) # 687 missing - need to replace empty string values

# Convert empty strings to NA in Cabin
train$Cabin[train$Cabin == ""] <- NA

# Verify the NA values
any_na(train$Cabin)  # Should now return TRUE
describe(train$Cabin)  # Should remain consistent
n_miss(train$Cabin) # 687 : issue solved

# Now this will work correctly:
train$HasCabin <- ifelse(!is.na(train$Cabin), 1, 0)
# describe(train$HasCabin) - perfect





