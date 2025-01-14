---
title: "Customer Analytics - Preparing Data for Modeling"
author: "Axl Ibiza, MBA"
date: "2024-11-19"
---

# Load necessary libraries
library(readr)
library(dplyr)
library(forcats)

# Load the dataset
ds_jobs <- read_csv("customer_train.csv")

# Clean up and categorize the data, handling missing values and collapsing company_size levels
ds_jobs_clean <- ds_jobs %>%
  mutate(
    student_id = as.integer(student_id),
    city = as.factor(city),
    city_development_index = as.double(city_development_index),
    gender = as.factor(gender),
    relevant_experience = as.factor(relevant_experience),
    enrolled_university = as.factor(enrolled_university),
    education_level = as.factor(education_level),
    major_discipline = as.factor(major_discipline),
    experience = case_when(
      experience %in% c("<1", "1", "2", "3", "4") ~ "<5",
      experience %in% c("5", "6", "7", "8", "9", "10") ~ "5-10",
      experience %in% c("11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20") ~ ">10",
      TRUE ~ experience
    ) %>% as.factor(),
    
    # Fix the collapsing of company_size levels
    company_size = case_when(
      company_size %in% c("<10") ~ "Micro", 
      company_size %in% c("10-49", "50-99") ~ "Small",
      company_size %in% c("100-499", "500-999") ~ "Medium",
      company_size %in% c("1000-4999", "5000-9999", "10000+", "1000+") ~ "Large",
      is.na(company_size) | company_size == "null" ~ NA_character_,  # Handle missing/null values
      TRUE ~ company_size
    ) %>% as.factor(),
    
    company_type = as.factor(company_type),
    last_new_job = as.factor(last_new_job),
    training_hours = as.integer(training_hours),
    job_change = as.integer(job_change)
  )

# Filter out rows with missing values in company_size and experience, then apply the filter for Large companies and >10 years experience
ds_jobs_clean <- ds_jobs_clean %>%
  filter(!is.na(company_size), experience == ">10" & company_size == "Large")

# Display the structure and types of the cleaned data
str(ds_jobs_clean)

# Check the levels of company_size to ensure they are correctly collapsed
levels(ds_jobs_clean$company_size)

# Check memory usage of the original and cleaned data frames
print(object.size(ds_jobs))
print(object.size(ds_jobs_clean))