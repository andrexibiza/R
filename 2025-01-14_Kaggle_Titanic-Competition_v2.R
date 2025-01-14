
# ---
# Title: "Titanic - Machine Learning from Disaster"
# Author: "Andrex Ibiza, MBA"
# Date: 2025-01-14
# Version: 2
# Score: "TBD"
# ---
  
# Load packages
library(dplyr)        # data manipulation
library(tidyr)        # data cleaning
library(randomForest) # inference model
library(caret)        # machine learning
library(naniar)       # for working with missing data
library(Hmisc)        # robust describe() function
library(ggplot2)      # viz

# Load `train.csv` data
train <- read.csv("train.csv", stringsAsFactors = FALSE)
head(train, 5) #--loaded successfully

# Evaluate structure and data types
str(train)

# 'data.frame':	891 obs. of  12 variables:
#   $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
# $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
# $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
# $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
# $ Sex        : chr  "male" "female" "female" "female" ...
# $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
# $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
# $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
# $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
# $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
# $ Cabin      : chr  "" "C85" "" "C123" ...
# $ Embarked   : chr  "S" "C" "S" "S"

# These data need to be prepared for ingestion by a random forest model,
# which calls for standard preprocessing techniques like converting data types,
# centering, scaling


# describe(train)

# train 
# 
# 12  Variables      891  Observations
# --------------------------------------------------------------------------------------------------------------------------------------------
#   PassengerId 
# n  missing distinct     Info     Mean  pMedian      Gmd      .05      .10      .25      .50      .75      .90      .95 
# 891        0      891        1      446      446    297.3     45.5     90.0    223.5    446.0    668.5    802.0    846.5 
# 
# lowest :   1   2   3   4   5, highest: 887 888 889 890 891
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Survived 
# n  missing distinct     Info      Sum     Mean 
# 891        0        2     0.71      342   0.3838 
# 
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Pclass 
# n  missing distinct     Info     Mean  pMedian      Gmd 
# 891        0        3     0.81    2.309      2.5   0.8631 
# 
# Value          1     2     3
# Frequency    216   184   491
# Proportion 0.242 0.207 0.551
# 
# For the frequency table, variable is rounded to the nearest 0
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Name 
# n  missing distinct 
# 891        0      891 
# 
# lowest : Abbing, Mr. Anthony                    Abbott, Mr. Rossmore Edward            Abbott, Mrs. Stanton (Rosa Hunt)       Abelson, Mr. Samuel                    Abelson, Mrs. Samuel (Hannah Wizosky) 
# highest: Yousseff, Mr. Gerious                  Yrois, Miss. Henriette ("Mrs Harbeck") Zabour, Miss. Hileni                   Zabour, Miss. Thamine                  Zimmerman, Mr. Leo                    
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Sex 
# n  missing distinct 
# 891        0        2 
# 
# Value      female   male
# Frequency     314    577
# Proportion  0.352  0.648
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Age 
# n  missing distinct     Info     Mean  pMedian      Gmd      .05      .10      .25      .50      .75      .90      .95 
# 714      177       88    0.999     29.7       29    16.21     4.00    14.00    20.12    28.00    38.00    50.00    56.00 
# 
# lowest : 0.42 0.67 0.75 0.83 0.92, highest: 70   70.5 71   74   80  
# --------------------------------------------------------------------------------------------------------------------------------------------
#   SibSp 
# n  missing distinct     Info     Mean  pMedian      Gmd 
# 891        0        7    0.669    0.523      0.5    0.823 
# 
# Value          0     1     2     3     4     5     8
# Frequency    608   209    28    16    18     5     7
# Proportion 0.682 0.235 0.031 0.018 0.020 0.006 0.008
# 
# For the frequency table, variable is rounded to the nearest 0
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Parch 
# n  missing distinct     Info     Mean  pMedian      Gmd 
# 891        0        7    0.556   0.3816        0   0.6259 
# 
# Value          0     1     2     3     4     5     6
# Frequency    678   118    80     5     4     5     1
# Proportion 0.761 0.132 0.090 0.006 0.004 0.006 0.001
# 
# For the frequency table, variable is rounded to the nearest 0
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Ticket 
# n  missing distinct 
# 891        0      681 
# 
# lowest : 110152      110413      110465      110564      110813     , highest: W./C. 6608  W./C. 6609  W.E.P. 5734 W/C 14208   WE/P 5735  
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Fare 
# n  missing distinct     Info     Mean  pMedian      Gmd      .05      .10      .25      .50      .75      .90      .95 
# 891        0      248        1     32.2     19.6    36.78    7.225    7.550    7.910   14.454   31.000   77.958  112.079 
# 
# lowest : 0       4.0125  5       6.2375  6.4375 , highest: 227.525 247.521 262.375 263     512.329
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Cabin 
# n  missing distinct 
# 204      687      147 
# 
# lowest : A10 A14 A16 A19 A20, highest: F33 F38 F4  G6  T  
# --------------------------------------------------------------------------------------------------------------------------------------------
#   Embarked 
# n  missing distinct 
# 889        2        3 
# 
# Value          C     Q     S
# Frequency    168    77   644
# Proportion 0.189 0.087 0.724
# --------------------------------------------------------------------------------------------------------------------------------------------
  
# # Data Cleaning
# 
# ## Missing Values
# Preparing the data for modeling requires addressing missing values in the dataset. 
# * `Age`: 177 missing values. We need to utilize a Random Forest model to impute missing age values before modeling.
# * `Cabin`: 687 missing values. There are too many missing values to impute them. This column will be converted to a binary operator of 1 if a cabin was recorded and 0 if not.
# * `Embarked`: 2 missing values. These will be imputed with the mode, since only two are missing.
# 
# ## Feature Engineering
# * `HasCabin`: 0 if `Cabin` entry missing, 1 if complete.
# * `Sex`: encode as binary variable based on these data (0 = female, 1 = male) to make this variable easier to model.
# * `Embarked`: convert to factors, then to numeric.





#--Predict missing `Age` values using other features
#--do this last since the model relies on the rest of the data already being as clean as possible.
# Prepare data for age prediction
age_data <- train %>%
  select(Age, Pclass, Sex, SibSp, Parch, Fare, Embarked)

# Separate rows with and without missing Age
age_complete <- age_data %>% filter(!is.na(Age))
age_missing <- age_data %>% filter(is.na(Age))


# Train a Random Forest model to predict Age
set.seed(666)
age_model <- randomForest(
  Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
  data = age_complete,
  ntree = 100
)

# Predict missing ages
predicted_ages <- predict(age_model, newdata = age_missing)
train$Age[is.na(train$Age)] <- predicted_ages

describe(train)

# 1. Compute the mode of the original Embarked column (ignoring NA values)
embarked_mode <- names(sort(table(train$Embarked[!is.na(train$Embarked)]), decreasing = TRUE))[1]

# 2. Impute missing Embarked values with the mode
train$Embarked[is.na(train$Embarked)] <- embarked_mode

# 3. Convert Embarked to numeric factors
# Levels are explicitly defined for consistency in encoding
train$Embarked <- as.numeric(
  factor(
    train$Embarked, 
    levels = c("C", "Q", "S"),  # Original categories
    labels = c(1, 2, 3)         # Numeric encoding
  )
)

# Convert Sex to binary numeric factor
train$Sex <- as.numeric(
  factor(
    train$Sex, 
    levels = c("male", "female"), 
    labels = c(1, 0)
  )
)

# Step 4: Engineer binary HasCabin feature
train$HasCabin <- ifelse(!is.na(train$Cabin), 1, 0)

# Step 5: Standardize Numerical Features
library(caret)
pre_process <- preProcess(train[, c("Fare", "Age", "SibSp", "Parch")], method = c("center", "scale"))
train[, c("Fare", "Age", "SibSp", "Parch")] <- predict(pre_process, train[, c("Fare", "Age", "SibSp", "Parch")])

# Step 6: Save cleaned dataset
# write.csv(train, "train_clean_v2.csv", row.names = FALSE)

# Step 7: Check cleaned data
library(Hmisc)
describe(train)
