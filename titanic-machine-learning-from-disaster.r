{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0cc3eec3",
   "metadata": {
    "papermill": {
     "duration": 0.007136,
     "end_time": "2025-01-18T20:50:42.415359",
     "exception": false,
     "start_time": "2025-01-18T20:50:42.408223",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Titanic - Machine Learning from Disaster\n",
    "\n",
    "**Andrex Ibiza, MBA**\n",
    "2025-01-16"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d21159d",
   "metadata": {
    "papermill": {
     "duration": 0.003857,
     "end_time": "2025-01-18T20:50:42.425437",
     "exception": false,
     "start_time": "2025-01-18T20:50:42.421580",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Introduction\n",
    "\n",
    "This notebook documents my second attempt at working through the Titanic dataset to build an accurate predictive model for Titanic shipwreck survivors (https://www.kaggle.com/competitions/titanic). My v1 model scored around 70% accuracy. In this iteration, to build a more accurate model, I plan to take a more nuanced approach toward fully exploring the data, dealing with missing values, and engineering meaningful new features.\n",
    "\n",
    "## Files\n",
    "* `gender_submission.csv`: example of what the final submitted file should look like with two columns: `PassengerID` and `Survived`.\n",
    "* `train.csv`: labeled data (`Survived`) used to build the model. 11 columns\n",
    "* `test.csv`: 12 columns\n",
    "\n",
    "## Data dictionary\n",
    "| Variable\t| Definition | Key | Notes |\n",
    "| --- | --- | --- | --- |\n",
    "| survival\t| Survival\t| 0 = No, 1 = Yes | --- |\n",
    "| pclass\t| Ticket class\t| 1 = 1st, 2 = 2nd, 3 = 3rd | Proxy for SES- 1st=upper, 2nd=middle, 3rd=lower |\n",
    "| sex\t| Sex | --- | --- |\n",
    "| Age\t| Age in years | --- | Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5 |\n",
    "| sibsp\t| # of siblings / spouses aboard the Titanic | --- | Sibling = brother, sister, stepbrother, stepsister; Spouse = husband, wife (mistresses and fiancés were ignored) |\n",
    "| parch\t| # of parents / children aboard the Titanic | --- | Parent = mother/father, Spouse = husband, wife (mistresses and fiances ignored). Some children travelled only with a nanny, therefore parch=0 for them. |\n",
    "| ticket | Ticket number | --- | --- |\n",
    "| fare\t| Passenger fare | --- | --- |\n",
    "| cabin\t| Cabin number | --- | --- |\n",
    "| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton | --- ||mpton | --- |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0289ff9",
   "metadata": {
    "papermill": {
     "duration": 0.004204,
     "end_time": "2025-01-18T20:50:42.433314",
     "exception": false,
     "start_time": "2025-01-18T20:50:42.429110",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Exploratory Data Analysis\n",
    "\n",
    "The first step in working with this dataset is to load `test.csv` into a dataframe to check its structure, data types, and identify any missing values. The `Hmisc` package provides a robust `describe()` function that provides detailed summary statistics for each variable in a dataset and helps identify missing values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f5a0b916",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:42.452194Z",
     "iopub.status.busy": "2025-01-18T20:50:42.449506Z",
     "iopub.status.idle": "2025-01-18T20:50:46.073974Z",
     "shell.execute_reply": "2025-01-18T20:50:46.072062Z"
    },
    "papermill": {
     "duration": 3.638071,
     "end_time": "2025-01-18T20:50:46.077134",
     "exception": false,
     "start_time": "2025-01-18T20:50:42.439063",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Loading required package: ggplot2\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Loading required package: lattice\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "Attaching package: ‘caret’\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following object is masked from ‘package:httr’:\n",
      "\n",
      "    progress\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "Attaching package: ‘dplyr’\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following objects are masked from ‘package:stats’:\n",
      "\n",
      "    filter, lag\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following objects are masked from ‘package:base’:\n",
      "\n",
      "    intersect, setdiff, setequal, union\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "Attaching package: ‘Hmisc’\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following objects are masked from ‘package:dplyr’:\n",
      "\n",
      "    src, summarize\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following objects are masked from ‘package:base’:\n",
      "\n",
      "    format.pval, units\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "randomForest 4.7-1.1\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Type rfNews() to see new features/changes/bug fixes.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "Attaching package: ‘randomForest’\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following object is masked from ‘package:dplyr’:\n",
      "\n",
      "    combine\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following object is masked from ‘package:ggplot2’:\n",
      "\n",
      "    margin\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 12</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>PassengerId</th><th scope=col>Survived</th><th scope=col>Pclass</th><th scope=col>Name</th><th scope=col>Sex</th><th scope=col>Age</th><th scope=col>SibSp</th><th scope=col>Parch</th><th scope=col>Ticket</th><th scope=col>Fare</th><th scope=col>Cabin</th><th scope=col>Embarked</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris                            </td><td>male  </td><td>22</td><td>1</td><td>0</td><td>A/5 21171       </td><td> 7.2500</td><td>    </td><td>S</td></tr>\n",
       "\t<tr><th scope=row>2</th><td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td>female</td><td>38</td><td>1</td><td>0</td><td>PC 17599        </td><td>71.2833</td><td>C85 </td><td>C</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina                             </td><td>female</td><td>26</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td> 7.9250</td><td>    </td><td>S</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)       </td><td>female</td><td>35</td><td>1</td><td>0</td><td>113803          </td><td>53.1000</td><td>C123</td><td>S</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>5</td><td>0</td><td>3</td><td>Allen, Mr. William Henry                           </td><td>male  </td><td>35</td><td>0</td><td>0</td><td>373450          </td><td> 8.0500</td><td>    </td><td>S</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>6</td><td>0</td><td>3</td><td>Moran, Mr. James                                   </td><td>male  </td><td>NA</td><td>0</td><td>0</td><td>330877          </td><td> 8.4583</td><td>    </td><td>Q</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 12\n",
       "\\begin{tabular}{r|llllllllllll}\n",
       "  & PassengerId & Survived & Pclass & Name & Sex & Age & SibSp & Parch & Ticket & Fare & Cabin & Embarked\\\\\n",
       "  & <int> & <int> & <int> & <chr> & <chr> & <dbl> & <int> & <int> & <chr> & <dbl> & <chr> & <chr>\\\\\n",
       "\\hline\n",
       "\t1 & 1 & 0 & 3 & Braund, Mr. Owen Harris                             & male   & 22 & 1 & 0 & A/5 21171        &  7.2500 &      & S\\\\\n",
       "\t2 & 2 & 1 & 1 & Cumings, Mrs. John Bradley (Florence Briggs Thayer) & female & 38 & 1 & 0 & PC 17599         & 71.2833 & C85  & C\\\\\n",
       "\t3 & 3 & 1 & 3 & Heikkinen, Miss. Laina                              & female & 26 & 0 & 0 & STON/O2. 3101282 &  7.9250 &      & S\\\\\n",
       "\t4 & 4 & 1 & 1 & Futrelle, Mrs. Jacques Heath (Lily May Peel)        & female & 35 & 1 & 0 & 113803           & 53.1000 & C123 & S\\\\\n",
       "\t5 & 5 & 0 & 3 & Allen, Mr. William Henry                            & male   & 35 & 0 & 0 & 373450           &  8.0500 &      & S\\\\\n",
       "\t6 & 6 & 0 & 3 & Moran, Mr. James                                    & male   & NA & 0 & 0 & 330877           &  8.4583 &      & Q\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 12\n",
       "\n",
       "| <!--/--> | PassengerId &lt;int&gt; | Survived &lt;int&gt; | Pclass &lt;int&gt; | Name &lt;chr&gt; | Sex &lt;chr&gt; | Age &lt;dbl&gt; | SibSp &lt;int&gt; | Parch &lt;int&gt; | Ticket &lt;chr&gt; | Fare &lt;dbl&gt; | Cabin &lt;chr&gt; | Embarked &lt;chr&gt; |\n",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
       "| 1 | 1 | 0 | 3 | Braund, Mr. Owen Harris                             | male   | 22 | 1 | 0 | A/5 21171        |  7.2500 | <!----> | S |\n",
       "| 2 | 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38 | 1 | 0 | PC 17599         | 71.2833 | C85  | C |\n",
       "| 3 | 3 | 1 | 3 | Heikkinen, Miss. Laina                              | female | 26 | 0 | 0 | STON/O2. 3101282 |  7.9250 | <!----> | S |\n",
       "| 4 | 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female | 35 | 1 | 0 | 113803           | 53.1000 | C123 | S |\n",
       "| 5 | 5 | 0 | 3 | Allen, Mr. William Henry                            | male   | 35 | 0 | 0 | 373450           |  8.0500 | <!----> | S |\n",
       "| 6 | 6 | 0 | 3 | Moran, Mr. James                                    | male   | NA | 0 | 0 | 330877           |  8.4583 | <!----> | Q |\n",
       "\n"
      ],
      "text/plain": [
       "  PassengerId Survived Pclass\n",
       "1 1           0        3     \n",
       "2 2           1        1     \n",
       "3 3           1        3     \n",
       "4 4           1        1     \n",
       "5 5           0        3     \n",
       "6 6           0        3     \n",
       "  Name                                                Sex    Age SibSp Parch\n",
       "1 Braund, Mr. Owen Harris                             male   22  1     0    \n",
       "2 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female 38  1     0    \n",
       "3 Heikkinen, Miss. Laina                              female 26  0     0    \n",
       "4 Futrelle, Mrs. Jacques Heath (Lily May Peel)        female 35  1     0    \n",
       "5 Allen, Mr. William Henry                            male   35  0     0    \n",
       "6 Moran, Mr. James                                    male   NA  0     0    \n",
       "  Ticket           Fare    Cabin Embarked\n",
       "1 A/5 21171         7.2500       S       \n",
       "2 PC 17599         71.2833 C85   C       \n",
       "3 STON/O2. 3101282  7.9250       S       \n",
       "4 113803           53.1000 C123  S       \n",
       "5 373450            8.0500       S       \n",
       "6 330877            8.4583       Q       "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 11</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>PassengerId</th><th scope=col>Pclass</th><th scope=col>Name</th><th scope=col>Sex</th><th scope=col>Age</th><th scope=col>SibSp</th><th scope=col>Parch</th><th scope=col>Ticket</th><th scope=col>Fare</th><th scope=col>Cabin</th><th scope=col>Embarked</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>892</td><td>3</td><td>Kelly, Mr. James                            </td><td>male  </td><td>34.5</td><td>0</td><td>0</td><td>330911 </td><td> 7.8292</td><td></td><td>Q</td></tr>\n",
       "\t<tr><th scope=row>2</th><td>893</td><td>3</td><td>Wilkes, Mrs. James (Ellen Needs)            </td><td>female</td><td>47.0</td><td>1</td><td>0</td><td>363272 </td><td> 7.0000</td><td></td><td>S</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>894</td><td>2</td><td>Myles, Mr. Thomas Francis                   </td><td>male  </td><td>62.0</td><td>0</td><td>0</td><td>240276 </td><td> 9.6875</td><td></td><td>Q</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>895</td><td>3</td><td>Wirz, Mr. Albert                            </td><td>male  </td><td>27.0</td><td>0</td><td>0</td><td>315154 </td><td> 8.6625</td><td></td><td>S</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>896</td><td>3</td><td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td><td>female</td><td>22.0</td><td>1</td><td>1</td><td>3101298</td><td>12.2875</td><td></td><td>S</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>897</td><td>3</td><td>Svensson, Mr. Johan Cervin                  </td><td>male  </td><td>14.0</td><td>0</td><td>0</td><td>7538   </td><td> 9.2250</td><td></td><td>S</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 11\n",
       "\\begin{tabular}{r|lllllllllll}\n",
       "  & PassengerId & Pclass & Name & Sex & Age & SibSp & Parch & Ticket & Fare & Cabin & Embarked\\\\\n",
       "  & <int> & <int> & <chr> & <chr> & <dbl> & <int> & <int> & <chr> & <dbl> & <chr> & <chr>\\\\\n",
       "\\hline\n",
       "\t1 & 892 & 3 & Kelly, Mr. James                             & male   & 34.5 & 0 & 0 & 330911  &  7.8292 &  & Q\\\\\n",
       "\t2 & 893 & 3 & Wilkes, Mrs. James (Ellen Needs)             & female & 47.0 & 1 & 0 & 363272  &  7.0000 &  & S\\\\\n",
       "\t3 & 894 & 2 & Myles, Mr. Thomas Francis                    & male   & 62.0 & 0 & 0 & 240276  &  9.6875 &  & Q\\\\\n",
       "\t4 & 895 & 3 & Wirz, Mr. Albert                             & male   & 27.0 & 0 & 0 & 315154  &  8.6625 &  & S\\\\\n",
       "\t5 & 896 & 3 & Hirvonen, Mrs. Alexander (Helga E Lindqvist) & female & 22.0 & 1 & 1 & 3101298 & 12.2875 &  & S\\\\\n",
       "\t6 & 897 & 3 & Svensson, Mr. Johan Cervin                   & male   & 14.0 & 0 & 0 & 7538    &  9.2250 &  & S\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 11\n",
       "\n",
       "| <!--/--> | PassengerId &lt;int&gt; | Pclass &lt;int&gt; | Name &lt;chr&gt; | Sex &lt;chr&gt; | Age &lt;dbl&gt; | SibSp &lt;int&gt; | Parch &lt;int&gt; | Ticket &lt;chr&gt; | Fare &lt;dbl&gt; | Cabin &lt;chr&gt; | Embarked &lt;chr&gt; |\n",
       "|---|---|---|---|---|---|---|---|---|---|---|---|\n",
       "| 1 | 892 | 3 | Kelly, Mr. James                             | male   | 34.5 | 0 | 0 | 330911  |  7.8292 | <!----> | Q |\n",
       "| 2 | 893 | 3 | Wilkes, Mrs. James (Ellen Needs)             | female | 47.0 | 1 | 0 | 363272  |  7.0000 | <!----> | S |\n",
       "| 3 | 894 | 2 | Myles, Mr. Thomas Francis                    | male   | 62.0 | 0 | 0 | 240276  |  9.6875 | <!----> | Q |\n",
       "| 4 | 895 | 3 | Wirz, Mr. Albert                             | male   | 27.0 | 0 | 0 | 315154  |  8.6625 | <!----> | S |\n",
       "| 5 | 896 | 3 | Hirvonen, Mrs. Alexander (Helga E Lindqvist) | female | 22.0 | 1 | 1 | 3101298 | 12.2875 | <!----> | S |\n",
       "| 6 | 897 | 3 | Svensson, Mr. Johan Cervin                   | male   | 14.0 | 0 | 0 | 7538    |  9.2250 | <!----> | S |\n",
       "\n"
      ],
      "text/plain": [
       "  PassengerId Pclass Name                                         Sex    Age \n",
       "1 892         3      Kelly, Mr. James                             male   34.5\n",
       "2 893         3      Wilkes, Mrs. James (Ellen Needs)             female 47.0\n",
       "3 894         2      Myles, Mr. Thomas Francis                    male   62.0\n",
       "4 895         3      Wirz, Mr. Albert                             male   27.0\n",
       "5 896         3      Hirvonen, Mrs. Alexander (Helga E Lindqvist) female 22.0\n",
       "6 897         3      Svensson, Mr. Johan Cervin                   male   14.0\n",
       "  SibSp Parch Ticket  Fare    Cabin Embarked\n",
       "1 0     0     330911   7.8292       Q       \n",
       "2 1     0     363272   7.0000       S       \n",
       "3 0     0     240276   9.6875       Q       \n",
       "4 0     0     315154   8.6625       S       \n",
       "5 1     1     3101298 12.2875       S       \n",
       "6 0     0     7538     9.2250       S       "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Load packages\n",
    "library(caret)        # machine learning\n",
    "library(dplyr)        # data manipulation\n",
    "library(ggplot2)      # viz\n",
    "library(Hmisc)        # robust describe() function\n",
    "library(naniar)       # working with missing data\n",
    "library(randomForest) # inference model\n",
    "\n",
    "# Load train and test data\n",
    "train <- read.csv(\"/kaggle/input/titanic/train.csv\", stringsAsFactors = FALSE)\n",
    "test <- read.csv(\"/kaggle/input/titanic/test.csv\", stringsAsFactors = FALSE)\n",
    "head(train) #--loaded successfully\n",
    "head(test)  #--loaded successfully\n",
    "\n",
    "# Evaluate structure and data types\n",
    "# str(train)\n",
    "# str(test)\n",
    "# \n",
    "# describe(train)\n",
    "# train has missing values: Age 177, Cabin 687, Embarked 2\n",
    "# describe(test)\n",
    "# test has missing values: Cabin 327, Fare 1, Age 86"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5ef1e1f",
   "metadata": {
    "papermill": {
     "duration": 0.005023,
     "end_time": "2025-01-18T20:50:46.088145",
     "exception": false,
     "start_time": "2025-01-18T20:50:46.083122",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Data Cleaning and Preprocessing\n",
    "\n",
    "## 1) Encode Categorical Variables\n",
    "We need to encode the categorical variables correctly before using these variables to impute missing `Age` values with a random forest model.\n",
    "* `Sex`: Binary encode (male = 0, female = 1).\n",
    "* `Pclass`: Ordinal encode (1 = 1st class, 2 = 2nd class, 3 = 3rd class).\n",
    "* `Embarked`: One-hot encode (C, Q, S).\n",
    "\n",
    "## 2) Data Transformation\n",
    "* `Fare`: Highly skewed (95th percentile = 112.08, max = 512.33). Apply a log transformation (log(Fare + 1)) to reduce skew.\n",
    "\n",
    "## 3) Missing Values\n",
    "Preparing the data for modeling requires addressing missing values in the dataset. \n",
    "* `Age`: 177 missing values. We will apply a random forest model to impute missing ages, instead of simpler imputation methods like median or mode. Perform cross-validation to estimate how well the model predicts Age for rows with non-missing values.\n",
    "* `Cabin`: 687 missing values. There are too many missing values to impute them. This column will be converted to a new binary column called `HasCabin` of 1 if a cabin was recorded and 0 if not.\n",
    "* `Embarked`: 2 missing values. These will be imputed with the mode, since only two are missing.\n",
    "\n",
    "## 4) Feature Engineering\n",
    "* `HasCabin`: 0 if `Cabin` entry missing, 1 if complete.\n",
    "* `SibSp` and `Parch`: Combine into a new `FamilySize = SibSp + Parch + 1`. Family size may capture survival trends better than the individual components.\n",
    "\n",
    "## 5) Remove Unnecessary Features\n",
    "* `Cabin`: after extracting `HasCabin` feature.\n",
    "* `Name`: We could consider extracting titles (`Mr.`, `Mrs.`, `Miss`, etc.) as a new feature. Titles may capture social status or age-related trends. For this iteration, we will drop the `Name` variable entirely without adding new features.\n",
    "* `PassengerId`: purely an identifier\n",
    "* `Ticket`: although there could potentially be useful patterns in the ticket prefixes, we will drop this column for this iteration since the data seem noisy.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0234628c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:46.148403Z",
     "iopub.status.busy": "2025-01-18T20:50:46.100697Z",
     "iopub.status.idle": "2025-01-18T20:50:46.223616Z",
     "shell.execute_reply": "2025-01-18T20:50:46.221891Z"
    },
    "papermill": {
     "duration": 0.133196,
     "end_time": "2025-01-18T20:50:46.226746",
     "exception": false,
     "start_time": "2025-01-18T20:50:46.093550",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in train$Embarked[is.na(train$Embarked)] <- embarked_mode:\n",
      "“number of items to replace is not a multiple of replacement length”\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "train$Embarked \n",
       "       n  missing distinct \n",
       "     891        0        3 \n",
       "                            \n",
       "Value          C     Q     S\n",
       "Frequency    169    78   644\n",
       "Proportion 0.190 0.088 0.723"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# DATA CLEANING AND PREPROCESSING\n",
    "\n",
    "# 1) Encode categorical variables\n",
    "# [X] Encode Sex as numeric factor\n",
    "train$Sex <- ifelse(train$Sex == \"male\", 1, 0)\n",
    "test$Sex <- ifelse(test$Sex == \"male\", 1, 0)\n",
    "# head(train[, \"Sex\"]) #--encoded successfully\n",
    "# head(test[, \"Sex\"]) #--encoded successfully\n",
    "\n",
    "# [X] Convert Pclass to an ordinal factor\n",
    "train$Pclass <- factor(train$Pclass, levels = c(1, 2, 3), ordered = TRUE)\n",
    "test$Pclass <- factor(test$Pclass, levels = c(1, 2, 3), ordered = TRUE)\n",
    "# head(train[, \"Pclass\"]) #--encoded successfully\n",
    "# head(test[, \"Pclass\"]) #--encoded successfully\n",
    "\n",
    "# [X] One-hot encode Embarked\n",
    "embarked_train_one_hot <- model.matrix(~ Embarked - 1, data = train)\n",
    "embarked_test_one_hot <- model.matrix(~ Embarked - 1, data = test)\n",
    "\n",
    "# Add the one-hot encoded columns back to the dataset\n",
    "train <- cbind(train, embarked_train_one_hot)\n",
    "test <- cbind(test, embarked_test_one_hot)\n",
    "\n",
    "# Verify encoding:\n",
    "# head(train[, c(\"Embarked\", \"EmbarkedC\", \"EmbarkedQ\", \"EmbarkedS\")])\n",
    "# head(test[, c(\"Embarked\", \"EmbarkedC\", \"EmbarkedQ\", \"EmbarkedS\")])\n",
    "\n",
    "# -- looks perfect, let's not forget about imputing our 2 missing values\n",
    "# Impute 2 missing Embarked values with the mode\n",
    "train$Embarked[train$Embarked == \"\"] <- NA\n",
    "embarked_mode <- names(sort(table(train$Embarked)))\n",
    "train$Embarked[is.na(train$Embarked)] <- embarked_mode\n",
    "\n",
    "# verify imputation\n",
    "describe(train$Embarked)\n",
    "\n",
    "# now drop the original Embarked column\n",
    "train <- train %>% select(-Embarked)\n",
    "test <- test %>% select(-Embarked)\n",
    "# str(train)\n",
    "# str(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c6aaeff2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:46.240445Z",
     "iopub.status.busy": "2025-01-18T20:50:46.238898Z",
     "iopub.status.idle": "2025-01-18T20:50:46.757092Z",
     "shell.execute_reply": "2025-01-18T20:50:46.754474Z"
    },
    "papermill": {
     "duration": 0.52808,
     "end_time": "2025-01-18T20:50:46.760082",
     "exception": false,
     "start_time": "2025-01-18T20:50:46.232002",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\nz88tNV/4jW98oPe6rajbNIQbFzzT/Ymu385uWexzMub9808hhFHrT17qG60YsXk+SVrn3VEM\nYbEKmHXvf0MIK0/psQCWejP7bqnvgu4lFR/9xEEf/cRBIYRiy+x7fvmDz3z2rFu/uc+Mk5qm\nLb/4xzx8mD0zyCrqPhJCaG34YwjnLLq8df6flrxwR/PTIYS68WsMymgADF0RPseuF2lx/s9n\nN5dVrr5oUoQQFjz31Pt/c1J+6qT6Yvvs0/+62ANepeM3mjBu3LjfzGnt9vuqx+6TT5I5f3ux\n27U3nbLYc8tKl53w/0II239lvaW+0XzVhOkr1hRaXjj1L/9ddHmh5bkvPfpWkqv48qRuHmH8\nkJvZRx/qLniv5tk/W3vttTfc4ktdS/LVK3z8kK9dvvboNE3vmtffe2Zwlddtsv/Ymrb5f772\nlcZFl8/85s+XvHDrW38KIay696qDNBwAQ9XwCrskP2Ktqnyx/ZXr/jmva+HffnnJzvv8LoRQ\nbHmf9xme/qNjQgjf3nmXGx9+vXNJWmz86ck7XfnErLaR/7PXmO4/0ClfNXHvMdVNr/2427X/\nvuXg46+9r/Pxv1Jh3vdP3O6S5xqql9/1inee3b90N3rmd/YIIVyx2163Pd3QuaTQNOurn9rh\nP22F1Xa9euqI8n7fzC5psbcn4X3Iu2BRVaM/3vDSv558+PKzfvNk18K3/vm7s/81P0nKpi/x\n5MVOH2bPDLJvXblPCOGUXY57ekFH55JZd1+6z7XPhRBC8p4j95Xf/D2E8MlPrjLYIwIw1Az6\n63AHUF/eoPjBs7YLIeTytVt/fI//2XvXjdZZMZevO+jU00II+Ypxh33+uOZiqfOtLra5/rkl\nv/2WU3bp3G9rbjh1px22mjC2KoRQOWqT23p9b9g/Hz4phHBfw3sGO37lurLK1T+2QnUIobJ+\nlY9+dMqoinwIoaxqzR8/Ne8D3eiSb+qRpqVLDt4ghJAk+VUnbbrtR9erK8uFEEZN3Ovp5sXf\nhnfJLV26zSx2vFWZS5Kk/BP7ffqzx9/d0/X35S5I33lPkI9d/fRit7JeTXm+Ylznvx869+Od\nc64wcaMdd97poxtOzCVJCGHn0+4ciD3T7Uh/3HutEMIRz81ddOEFa44KIfy+17c76X3TOl19\n6IYhhFz5iClTt91g/IohhE+d/70QwojVvrLoxS5ZZ3RZ9YTmXt5jEIDhYXidsQshbHnuH3/3\n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\nxWaloaGhqqqqqqoq60EG3GCE3YTpX7tkn7cfCEtLC7588jlbnX7BASuMqaofu3LF1Xc+MHvn\nT60WQuhoevzhxvZ9d14phFBVv0NPqwAA6NZghF3VimtMXPHtf3c+x65+jfHjV6oNIZy8/+Sv\nXH/O3eNOWX90x2+v/HbNuJ2mr1oXQghJRY+rAADoTsaf5TDxwPOPbbvsxkvPmtOaTNhou/PP\nOyrXh1UAACwpSVMf29pX06ZNG7TbmjFjxqDdVtw8xy5bnmM3FHiOXbY8x24oGD7PsXMWDAAg\nEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4A\nIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIO\nACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLC\nDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACAS\nwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAg\nEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4A\nIBLCDgAgEsIOACASwg4AIBLCDgAgEsIOACASwg4AIBJlWQ/QD0ql0ty5c7Oeop+99dZbWY8Q\nldbW1qxHGO7mzZuX9QjDXXNzc9YjDHdz5szJeoRhbeHChQsXLsx6in6Qz+dHjx7d09oYwi6X\ny9XX12c9RT+Lb4uy0tLSkqZpTU1N1oMMU8VisbGxMYQwcuTIXM5DBNlobm7O5XJVVVVZDzJM\ndXR0NDU1hRBGjRqVJEnW4wxTjY2NlZWVFRUVWQ/SD3r/KYoh7EIIZWWRbEiX+LYoK7lcLk1T\n+zNz+Xw+n89nPcUwlSRJLpdzFGSlVCp1/qOsrEzYZWX4HAX+gAYAiISwAwCIhLADAIiEsAMA\niISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLAD\nAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISw\nAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiE\nsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCI\nhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMA\niISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLAD\nAIiEsAMAiISwAwCIRNng3Ez7gud+cPkPH/zHi6352tXXWm+/o4/bao26EEIIpXtvvOrW+x99\npTE/ecrUw044fHxN10i9rAIAYHGDc8YuvepLZz341krHnXHBN07/4uT8MxeffOpbHaUQwqyb\nz7j0poe22Peos0+cXvfiPaefdE3pne/pZRUAAEsajLBrm/+nP85u/uy5x265waS119/0iNO+\nUmx75aY3m0PafslNT0846LwDdt5y/c22+eJFxze9fucNrzaFEHpbBQBAdwYj7HJlY4844ojN\nR1S8/XVSFkKoyefa5t//cmtxl11W6VxcWb/1JnUVM+99I4TQyyoAALo1GM9aK6/dcO+9Nwwh\nzHv8r4++/vqj99y8/Pp7HLJCTctrT4QQ1qsp77rkujVldzwxPxwc2pt6XLWkNE3nz58/0Fsx\nyBoaGrIeIRKlUinYn9lJ07TzHwsWLEiSJNthhq1isZgkSXt7e9aDDFNdR0F8v6qWIcVisbm5\nubW1NetB+kEulxs5cmRPawf15Qj/feCPd7zw6ksvtWy575ohhFJbUwhhTNm7Zw3HlucLC1t7\nX7WkNE0LhcJADp6B+LYoW515R4aKxWLWIwxraZo6CjLnP/ZsRXMU5PP5XtYOathNPv6r/xtC\n82sPf+74C88dt94pk6tDCPMKpbp3RpzTUczXV4QQchU9rlpSkiRVVVWDsQGDKL4tykqhUEjT\ntLy8/P0vygBI07StrS2EUFlZ6YxdVjo6OpIkKSvzxgLZKJVKnadL/ceeofb29nw+33sSLSty\nud6eRzcYx/mCF/785xcrd//E1M4va1aeusdyVb+/843yzTYI4f5nWwqrVb69o59vKYzauj6E\nUF7b46olJUlSV1c38NsxqOLboqw0NTWlaWp/ZqVQKHSGXU1NTRz/pS6LGhsb8/l8TU1N1oMM\n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\nbWlpGdhZ4zVMdt0w2cyhrLW1++OXQVMoFLIeYbjzH1G2SqVSR0dH1lP0g3w+P0TDLldRHUKY\nVyjV5fOdS+Z0FPP1Fb2vWlKSJPl3LsYHFf2uS9M0TdNczjv7ZKbzT+Tof9KGslKplCSJM6ZZ\nSdO089S1oyBDMR0Fvf9GyzLsyms3COH+Z1sKq1W+/bP+fEth1Nb1va9aUpIko0ePHpyZ4xP9\nrmtqakrTtK6uLutBhqlCodDQ0BBCGDlypN9qWWlsbMzn8zU1NVkPMky1t7cvWLAghFBfXx9H\nWCyLGhoaqqqqqqqq3v+iy7gsT2NU1e+wckX+zgdmd37Z0fT4w43tm+68Uu+rAADoVqaPTyUV\nJ+8/+YXrz7l75rOvz3ryurO+XTNup+mr1r3PKgAAupPxZzlMPPD8Y9suu/HSs+a0JhM22u78\n847K9WEVAABLSrwAvu+mTZuW9Qj9b8aMGVmPMLA8xy5bXc+xGz16tOfYZcVz7LLV9Ry7MWPG\neI5dVjzHDgCAZYywAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMA\niISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLAD\nAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISw\nAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiE\nsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCI\nhLADAIiEsAMAiERZ1gOQsWnTpmU9Qn+aMWNG1iMAQGacsQMAiISwAwCIhLADAIiEsAMAiISw\nAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIiEsAMAiISwAwCIhLADAIhE\nWdYD9I9isZj1CAwJS/4kpGmapqmfkKyUSqXOf7gLMuQoyNaiR0GSJNkOM2ylaVoqleI4CpIk\nyeV6PDEXQ9iVSqV58+ZlPQVDQk8/CW1tbYM8CYtZsGBB1iMMdy0tLVmPMNw1NDRkPcKw1tzc\n3NzcnPUU/SCfz48ePbqntTGEXS6XGzNmTNZTMCQs+ZPQ3NycpmltbW0m81AoFObPnx9CqK+v\n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"
     },
     "metadata": {
      "image/png": {
       "height": 420,
       "width": 420
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 2) Apply log transformation to Fare\n",
    "#--plot shape before transformation?\n",
    "ggplot(train, aes(x = Fare)) +\n",
    "  geom_histogram(bins=20) +\n",
    "  theme_minimal() +\n",
    "  ggtitle(\"Fare (before transforming)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9fac5d5a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:46.774814Z",
     "iopub.status.busy": "2025-01-18T20:50:46.773304Z",
     "iopub.status.idle": "2025-01-18T20:50:47.068459Z",
     "shell.execute_reply": "2025-01-18T20:50:47.066243Z"
    },
    "papermill": {
     "duration": 0.305059,
     "end_time": "2025-01-18T20:50:47.070819",
     "exception": false,
     "start_time": "2025-01-18T20:50:46.765760",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>\n",
       ".list-inline {list-style: none; margin:0; padding: 0}\n",
       ".list-inline>li {display: inline-block}\n",
       ".list-inline>li:not(:last-child)::after {content: \"\\00b7\"; padding: 0 .5ex}\n",
       "</style>\n",
       "<ol class=list-inline><li>2.11021320034659</li><li>4.2805931204649</li><li>2.1888563276657</li><li>3.99083418585244</li><li>2.20276475771183</li><li>2.24689266289817</li></ol>\n"
      ],
      "text/latex": [
       "\\begin{enumerate*}\n",
       "\\item 2.11021320034659\n",
       "\\item 4.2805931204649\n",
       "\\item 2.1888563276657\n",
       "\\item 3.99083418585244\n",
       "\\item 2.20276475771183\n",
       "\\item 2.24689266289817\n",
       "\\end{enumerate*}\n"
      ],
      "text/markdown": [
       "1. 2.11021320034659\n",
       "2. 4.2805931204649\n",
       "3. 2.1888563276657\n",
       "4. 3.99083418585244\n",
       "5. 2.20276475771183\n",
       "6. 2.24689266289817\n",
       "\n",
       "\n"
      ],
      "text/plain": [
       "[1] 2.110213 4.280593 2.188856 3.990834 2.202765 2.246893"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       ".list-inline {list-style: none; margin:0; padding: 0}\n",
       ".list-inline>li {display: inline-block}\n",
       ".list-inline>li:not(:last-child)::after {content: \"\\00b7\"; padding: 0 .5ex}\n",
       "</style>\n",
       "<ol class=list-inline><li>2.17806441028492</li><li>2.07944154167984</li><li>2.36907483426288</li><li>2.26825241391354</li><li>2.58682374366807</li><li>2.32483570192887</li></ol>\n"
      ],
      "text/latex": [
       "\\begin{enumerate*}\n",
       "\\item 2.17806441028492\n",
       "\\item 2.07944154167984\n",
       "\\item 2.36907483426288\n",
       "\\item 2.26825241391354\n",
       "\\item 2.58682374366807\n",
       "\\item 2.32483570192887\n",
       "\\end{enumerate*}\n"
      ],
      "text/markdown": [
       "1. 2.17806441028492\n",
       "2. 2.07944154167984\n",
       "3. 2.36907483426288\n",
       "4. 2.26825241391354\n",
       "5. 2.58682374366807\n",
       "6. 2.32483570192887\n",
       "\n",
       "\n"
      ],
      "text/plain": [
       "[1] 2.178064 2.079442 2.369075 2.268252 2.586824 2.324836"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nlCZ/fuKSCycMKE8pFVJKqXD+w2/lK4ZfuuvI932UXPlJh22QUvrvP76RUpr/yrfq2wp1Yy7Y\nb/iAJbcatc/31i36coDL/VVs66LnO9YWf9iX2fGidgEAelg/uo5dy4K/tGdZ7dD9ypc5lTN+\nzxHp4X+/9Pi8tPnw5vo/tmbZ4KF7LbVNdd1eKV28WiYZuvXQZRe2Nb5443U/+eNfH332uVkv\nvvTi/71Zv+w2KaW6yXWdf/Bz7v3xI1OOvvfOqz5651UVtWttuc12O++2x0GHH7PLxGHL3b7I\nw9LJ5Cmlssrl9F5NxfJ/bGhveuGFpraU3q5ewUm1+U/MTyktfP65lNKaO26/1NpcWc1hw2su\nf3XBct93ZRV52Jfa8SJ3AQB6WD8Ku5RW+BJSuXwupVRoKaSUskJTSimXlv6GncuttheNKB+w\n9GGf/bcfbLvbibMWtg4fv/Xu22+76/5HjNtok0lj7tt2u0uXO2onatf/2D1P//uh391y2x13\nz/zz/Q/N/PWDf7j9sq995WNn/eJXFy/3pF1Rh2VFk3dBlrWmlMqrNzjjtE8ud4O1t1szpZTr\nODO5vN0dtoJkXFnFH/aldrzIXQCAHtaPwq5y0Hb5XK5p7l3tKS3VaLPu+3dKaZ1JdSmlytoP\npZSa5v0+pfOW3Kap/g/dN9tJHzlt1sLWL8146NIj3vud5vwX/9rFD5er3ObDR2zz4SNSSu2L\n3rz3Fz/41GfOvf0bB8/4UsPUNQcstW2Rh2U1Kq8eu2ZFfk6h8eKvf72TSq3dYNOUfvfWAw+n\ntPTr/97T6avEFq/Lh73IXQCAHtaPnmOXrx579IiatkXPnfmXfy+5vG3RM1/+29u5ssrTJwxN\nKVXUbnno8Jrm+j9d98r7ftn3yDd+1k2DZe31P3uzsbxqvSXzIqU0/5knVvZDNb750/Hjx2+2\n/ZcXL8kPWGufo7565fihWZbdvbweKvKwrE65ijMn1LW3vDntr2++f0Xh5M3Hjhw58lezm1JK\ng0Z9aVhF2bznv3r37PeNPeefF8+sb171KVbpsBe3CwDQw/pR2KWUzrniYyml7+134B1PzutY\n0tYw6z/23+P/mttG73vNtoPeuY7GN686OKX0lSknPTm/tWPJrHsuO/i6Z1JKKbf6j1guP2jD\n6nx7yyvXPz538cKHfnHp3gf/OqXUvmglLnhbPXSfeS+98K8Hrzz3V/9avPDtx389/YX6XK78\n6BE1y32vIg/LanT0D09IKX1n7yk3Pfh6x5KsfcFPztjrqsdmNQ/+xIFrVKeU8lWjf3TEuKx9\n0Sd2PPqB/2vo2Gzuk3ceuMeFq2WGVTzsxewCAPS0Hv873O7ScbmTXH7Astctmzhx4sabbJ5l\nWZYVLj1yckopl8uPmrDVrttsUltellIaMu7AJxvfd4Hia47ZLKVUVjFo0ra7Th4zIqW0/4Xf\nTykNGv3/ihmm88ud7HLDM0stv//c3VJKZfmBO+/zsU8ctO/mG40oy9ceceZZHR/k2C+c1Nhe\n6LiwyI7XPLnU+25SU7HkDT3wtX067tm1xm2+5957bbPZuLJcLqW091m/7dhg2cudFHNYVjT5\ncqf6/UEbppQ+/cycJRdetMGQtMQ1Sn75lSkdc26w2bZ77bHT2OHVKaWqIVve8UbD4ndpa3rx\nExPrOgZbd6MtNx+3di6Xq6rb9opjx6fVcYHiYg77ina8yF0AgJ4ULexWJFc24N0N2+/90YUf\n3WnSsEEDyqsHrbfx9iece+2rzctcULbQevuVX9l3p82HVNWsu9EO51x//6I5d6SU6sZeXsww\nKxt2Wdb+6yvO3GHT9QZU5muHrrXjRz9162Ozsyz73jG7DakuH7jG6PltxYZdlmX/e+MlB+yy\n1ZpDBubLygcNW2fHfT551a2Pvrdnywm7Dz4sqz3ssix79LarDpuy7ZpDa8srqkeM2WzqFy96\nfN7SHdbe/Pr3v3r81uPXHVhZPmTNdfc76vRH5zT99bRJqyXsijnsnYRdkbsAAD0ml2Ur/KPI\nfmvOG68tas9GrLPuklcAmffc6UPHX7rhgffOunXP0o0GALBC/es5dkW6YddJo0aNunDW+y5p\n9sCFv04pbfuliSt4JwCAEhN2y3HItz6aUrp070//5pFZja3tDXNf+eWVpxz8k2er6nb93o5r\nl3o6AIDl86vY5cpuOG2/z1z5u8ISB2fgutv+4K67PjlpdV/7AwBgNRF2K/Tm4/f94jd/nPX6\nvMrBwzbeepeDPrrboA961QcAgBISdgAAQXiOHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAA\nghB2AABBCDsAgCCEXTcqFAr19fX19fVtbW2lnqVPamxsbGpqKvUUfVJLS0t9ff38+fNLPUhf\nNX/+/Pb29lJP0Sc1NjbW19c3NjaWepA+qa2tzcO2y+bPn19fX9/S0lLqQUqsvNQDBNfa2ppS\n8vIeXeM7a5cVCoXW1tZczovgdVFra2uhUMjn86UepO9pa2vzuddlWZZ1fNegC1pbW7Msq6qq\nKvUgJeaMHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAI\nYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh\n7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCDK\nSz0AUHpTp07tmRuaMWNGz9wQQP/kjB0AQBDCDgAgCGEHABCEsAMACELYAQAEIewAAIIQdgAA\nQQg7AIAghB0AQBDCDgAgCGEHABCEsAMACELYAQAEIewAAIIQdgAAQQg7AIAghB0AQBDCDgAg\nCGEHABCEsAMACELYAQAEIewAAIIQdgAAQQg7AIAghB0AQBDCDgAgCGEHABCEsAMACELYAQAE\nIewAAIIQdgAAQQg7AIAghB0AQBDCDgAgCGEHABCEsAMACELYAQAEIewAAIIQdgAAQQg7AIAg\nhB0AQBDCDgAgCGEHABCEsAMACELYAQAEIewAAIIQdgAAQQg7AIAghB0AQBDCDgAgCGEHABCE\nsAMACELYAQAEIewAAIIQdgAAQQg7AIAghB0AQBDCDgAgCGEHABCEsAMACELYAQAEIewAAIIQ\ndgAAQQg7AIAghB0AQBDCDgAgCGEHABCEsAMACELYAQAEIewAAIIoL/UAq0eWZaUeYTkWT5Vl\nWe+csE9w6Lpgyc+90k6ylN42Tyc8bFeRo9cFHQfNoVsV/eSRm8vlVrQqQtgVCoU5c+aUeorO\nzJ8/v9Qj9GGNjY2lHqGvyrJs9uzZpZ7ifXrbPJ3wsF0VLS0tfei+7m0culXR0NDQ0NBQ6im6\nVz6fHzp06IrWRgi7srKyTvawhAqFQn19fUpp0KBB5eURDnUPa2hoyOfz1dXVpR6k72lubm5s\nbMzlcnV1daWe5X1650N1WfPmzautrfWw7YKFCxe2trZWVFTU1taWepa+p62tbeHChb3tYdtX\nzJs3L8uympqaqqqqUs/SvTo5XZdihF1KKZ/Pl3qE5Vh86MvKynrnhL1cLpfL5XIOXReUlb3z\n9NnedvR62zyd8LDtmo6vex65XVMoFFKfepj0Qh65/ngCACAIYQcAEISwAwAIQtgBAAQh7AAA\nghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBA\nEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAI\nQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABB\nCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAI\nYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh\n7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCE\nHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISw\nAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2\nAABBCDsAgCCEHQBAEMIOACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIO\nACAIYQcAEISwAwAIQtgBAAQh7AAAghB2AABBCDsAgCCEHQBAEMIOACCI8p65maxt7i+vu/bO\n+/8xu6ls5OjxBxx1woe3XDullFLhvpuuvn3m315ZkJ84adtjTzluTM3ikTpZBQDA0nrojN3v\nLj7jxj/++4DjTv3mBWfuObb56vNOuvWVhSmlWbecfdnND2z/8eOnn3Z07fP3TvvStYV336WT\nVQAALKsnwq69+ZVrHnl7l3PO/dieO4yfuNkhJ108pS5/69X/SlnLpTc/OfaI8w/be4dNt97l\ni5ec3PD6b298tSGl1NkqAACWp0fCrunF9Tfc8CNjBr+7ILflkKrWeQub62e+3NQ+Zcq6HUur\n6nbesrbykfveSCl1sgoAgOXqiWetVQ7Z5fLLd1n8ZuvCp65/beH6x01oafh5SmmTmorFqzau\nKb/rsfp0ZGppeGxFq5ZVKBTmzp3bffOvuvnz55d6hD4py7KU0qJFi0o9SF+VZdns2bNLPcX7\n9LZ5ViTLMg/brul42La0tPSV+7q36YUP276i43OvoaGhoSH47/fy+XxdXd2K1vb0nyO89PAd\nV15xfeuY/abtO6rtpYaU0hrl7501HF6Rb1vYlFIqNK9w1XJ13J29Vi8fr5dz9FZFbzt6vW2e\nTvShUXsnB7DLHLpV0R+OXuf72HNh1zL36eu/e+Wdj87Z7dAvXDR1z+pcbkHlgJTS3LZCbT7f\nsc3s1vZ8XWVKqWzFq5aVy+UGDhzYE/uwkrIsa2xsTClVV1fn390Ritfc3FxWVlZRUfHBm/J+\nbW1tzc3NuVyupqam1LO8T+98qC6rsbGxqqrKw7YLmpqa2tvb8/l8dXV1qWfpe9rb25ubm3vb\nw7avaGxszLKsqqqqvDz4NTRyuVwna3to5xe8dO/pZ3wvP3m/S647esLwdx7tFQMnpzTz6UVt\no6ve+er57KK2ITvXdb5qWblcbsCAAd2/EyutUCh0hF1VVZU66YK2trZ8Pt8779xerqmpqbm5\nOaXU245eb5tnRTrCzsO2C1pbWzvCrq/c171Ka2trc3OzQ9c1Hd9wKyoq+vkPFT3xxxNZofGi\nM6+u2uvUq8/93OKqSylV1+2xTmX+t39+s+PN1oa/P7igZau91+58FQAAy9UTZ+wa37zxicbW\n4ybXPPLww+/d8IBxW2xad8ahE//fDefdM/Irmw5tve2q79SM3OvoUbUppZSrXOEqAACWpyfC\nbsFzL6aUfvjNi5ZcOHj0V3961fbjDr/wxObLb7rs3NlNubGb73bh+ccvPoXYySoAAJbVE2G3\n9s4X3bbzCtbl8lOOOX3KMSu5CgCAZTgLBgAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQd\nAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLAD\nAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYA\nAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4A\nIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEA\nBCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCA\nIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQ\nhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACC\nEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQ\nwg4AIAhhBwAQhLADAAhC2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQhLADAAhC\n2AEABCHsAACCEHYAAEEIOwCAIIQdAEAQwg4AIAhhBwAQRHmpB1gNsixraGgo9RTLkWVZx38W\nLVrU3Nxc2mH6ora2tvb29kKhUOpB+p729vaO/yxcuLC0kyylt82zIlmWedh2TVtbW8e/feW+\n7lUKhUKWZQ7dqmhubu74JAysrKyspqZmhWt7chQAALpPsWfsdthhh0N+fvcZo2qXWv7G/ace\ndvbcP/3+J6t7sJWQy+Vqa5cerDcoFAodP/EPGDCgoqKi1OP0PQsWLMjn8538XMKKNDU1tba2\nppR620Ojt82zIs3NzR62XTN//vyWlpby8vK+cl/3Kq2tra2trQ5d13R8w62qqqquri71LKX0\nAWE3/4XnXm9pTyn95S9/GfPkk083DH7/+uxfv5l5/59e7K7pAAAo2geE3S37bvfpZ+Z0/H/G\nPtvOWN42gzc4aXVPBQDASvuAsNvx/EuvmdeUUjrhhBN2u+CyI9YcsNQGZRWDdjjk0O6aDgCA\non1A2E04/JgJKaWUbrrppoM+/dnPr+MX/wAAvVSxfzzxhz/8oVvnAABgFa3cdezm/N+stxpa\nl10+YcKE1TQPAABdVGzYNb19zyE7H37H03OWu3bxlXgBACiVYsPuPw886s5nF+z/hbP23WyD\n8ly3jgQAQFcUG3YXPvTWmMP/5/arD+jWaQAA6LKiXlIsa1/wVmv7+odv1t3TAADQZUWFXS5f\nu3td9awbHu7uaQAA6LKiwi6l3E2/vqDlzk8de8GP/t3Q1r0TAQDQJcU+x+7Qs341YmTFj849\n9sfTPzNs7bUH5N/3BxSvvPJKN8wGAMBKKDbshg8fPnz43utv0a3DAADQdcWG3S9/+ctunQMA\ngFVUbNjV19d3snbIkCGrYxgAALqu2LCrq6vrZK1XngAAKLliw+68885739tZ22uznrj15l/N\nya173vcvXu1jAQCwsooNu+nTpy+78PJv/XWvjXa7/IpHph135GqdCgCAlVbkdeyWb8CI7a47\n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\nSUVORK5CYII="
     },
     "metadata": {
      "image/png": {
       "height": 420,
       "width": 420
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#--note an extreme outlier over 500!\n",
    "train$Fare <- log(train$Fare + 1)\n",
    "test$Fare <- log(test$Fare + 1)\n",
    "head(train[, \"Fare\"])\n",
    "head(test[, \"Fare\"])\n",
    "\n",
    "ggplot(train, aes(x = Fare)) +\n",
    "  geom_histogram(bins=20) +\n",
    "  theme_minimal() +\n",
    "  ggtitle(\"Log Transformed Fare\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6a10ea1b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:47.087004Z",
     "iopub.status.busy": "2025-01-18T20:50:47.085476Z",
     "iopub.status.idle": "2025-01-18T20:50:56.790142Z",
     "shell.execute_reply": "2025-01-18T20:50:56.788031Z"
    },
    "papermill": {
     "duration": 9.715601,
     "end_time": "2025-01-18T20:50:56.792769",
     "exception": false,
     "start_time": "2025-01-18T20:50:47.077168",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest \n",
      "\n",
      "714 samples\n",
      "  8 predictor\n",
      "\n",
      "No pre-processing\n",
      "Resampling: Cross-Validated (5 fold) \n",
      "Summary of sample sizes: 572, 570, 572, 571, 571 \n",
      "Resampling results across tuning parameters:\n",
      "\n",
      "  mtry  RMSE      Rsquared   MAE      \n",
      "  2     12.38531  0.2743195   9.693726\n",
      "  5     12.63533  0.2601041   9.806089\n",
      "  9     12.97427  0.2406396  10.027202\n",
      "\n",
      "RMSE was used to select the optimal model using the smallest value.\n",
      "The final value used for the model was mtry = 2.\n"
     ]
    }
   ],
   "source": [
    "# 3) Address missing values\n",
    "# Age - Train\n",
    "#--Predict missing ages using other features\n",
    "train_age_data <- train %>% \n",
    "    select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)\n",
    "\n",
    "# head(train[, c(\"Age\", \"Pclass\", \"Sex\", \"SibSp\", \"Parch\", \"Fare\", \"EmbarkedC\", \"EmbarkedQ\", \"EmbarkedS\")])\n",
    "#--verified that all these columns are formatted properly\n",
    "\n",
    "train_age_complete <- train_age_data %>% filter(!is.na(Age))\n",
    "train_age_missing <- train_age_data %>% filter(is.na(Age))\n",
    "\n",
    "set.seed(666)\n",
    "cv_control <- trainControl(method = \"cv\", number = 5)\n",
    "train_age_cv_model <- train(\n",
    "  Age ~ Pclass + Sex + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS,\n",
    "  data = train_age_complete,\n",
    "  method = \"rf\",\n",
    "  trControl = cv_control,\n",
    "  tuneLength = 3\n",
    ")\n",
    "print(train_age_cv_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3d67f6ac",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:56.809222Z",
     "iopub.status.busy": "2025-01-18T20:50:56.807674Z",
     "iopub.status.idle": "2025-01-18T20:50:56.865058Z",
     "shell.execute_reply": "2025-01-18T20:50:56.862762Z"
    },
    "papermill": {
     "duration": 0.068714,
     "end_time": "2025-01-18T20:50:56.868025",
     "exception": false,
     "start_time": "2025-01-18T20:50:56.799311",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "train$Age \n",
       "       n  missing distinct     Info     Mean      Gmd      .05      .10 \n",
       "     891        0      183        1     29.6    14.72     6.00    15.95 \n",
       "     .25      .50      .75      .90      .95 \n",
       "   21.19    28.58    36.00    47.00    54.00 \n",
       "\n",
       "lowest : 0.42 0.67 0.75 0.83 0.92, highest: 70   70.5 71   74   80  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Use the best model to predict missing ages\n",
    "predicted_train_ages <- predict(train_age_cv_model, newdata = train_age_missing)\n",
    "\n",
    "# Impute the predicted ages back into the train dataset\n",
    "train$Age[is.na(train$Age)] <- predicted_train_ages\n",
    "describe(train$Age)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cc304c3d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:56.884405Z",
     "iopub.status.busy": "2025-01-18T20:50:56.882854Z",
     "iopub.status.idle": "2025-01-18T20:50:56.923968Z",
     "shell.execute_reply": "2025-01-18T20:50:56.921549Z"
    },
    "papermill": {
     "duration": 0.052524,
     "end_time": "2025-01-18T20:50:56.926966",
     "exception": false,
     "start_time": "2025-01-18T20:50:56.874442",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "0"
      ],
      "text/latex": [
       "0"
      ],
      "text/markdown": [
       "0"
      ],
      "text/plain": [
       "[1] 0"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#--Age in test data\n",
    "# Preprocess the test data for Age imputation\n",
    "test_age_data <- test %>% \n",
    "  select(Age, Pclass, Sex, SibSp, Parch, Fare, EmbarkedC, EmbarkedQ, EmbarkedS)\n",
    "\n",
    "test_age_missing <- test_age_data %>% filter(is.na(Age))\n",
    "test_age_complete <- test_age_data %>% filter(!is.na(Age))\n",
    "\n",
    "# Use the trained train_age_cv_model to predict missing ages in the test dataset\n",
    "predicted_test_ages <- predict(train_age_cv_model, newdata = test_age_missing)\n",
    "\n",
    "# Impute the predicted ages back into the test dataset\n",
    "test$Age[is.na(test$Age)] <- predicted_test_ages\n",
    "\n",
    "n_miss(test$Age)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ace8693c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:56.944144Z",
     "iopub.status.busy": "2025-01-18T20:50:56.942552Z",
     "iopub.status.idle": "2025-01-18T20:50:57.001375Z",
     "shell.execute_reply": "2025-01-18T20:50:56.999655Z"
    },
    "papermill": {
     "duration": 0.069638,
     "end_time": "2025-01-18T20:50:57.003589",
     "exception": false,
     "start_time": "2025-01-18T20:50:56.933951",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 2</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>Cabin</th><th scope=col>HasCabin</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>NA  </td><td>0</td></tr>\n",
       "\t<tr><th scope=row>2</th><td>C85 </td><td>1</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>NA  </td><td>0</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>C123</td><td>1</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>NA  </td><td>0</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>NA  </td><td>0</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 2\n",
       "\\begin{tabular}{r|ll}\n",
       "  & Cabin & HasCabin\\\\\n",
       "  & <chr> & <dbl>\\\\\n",
       "\\hline\n",
       "\t1 & NA   & 0\\\\\n",
       "\t2 & C85  & 1\\\\\n",
       "\t3 & NA   & 0\\\\\n",
       "\t4 & C123 & 1\\\\\n",
       "\t5 & NA   & 0\\\\\n",
       "\t6 & NA   & 0\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 2\n",
       "\n",
       "| <!--/--> | Cabin &lt;chr&gt; | HasCabin &lt;dbl&gt; |\n",
       "|---|---|---|\n",
       "| 1 | NA   | 0 |\n",
       "| 2 | C85  | 1 |\n",
       "| 3 | NA   | 0 |\n",
       "| 4 | C123 | 1 |\n",
       "| 5 | NA   | 0 |\n",
       "| 6 | NA   | 0 |\n",
       "\n"
      ],
      "text/plain": [
       "  Cabin HasCabin\n",
       "1 NA    0       \n",
       "2 C85   1       \n",
       "3 NA    0       \n",
       "4 C123  1       \n",
       "5 NA    0       \n",
       "6 NA    0       "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 2</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>Cabin</th><th scope=col>HasCabin</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>NA</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>2</th><td>NA</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>NA</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>NA</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>NA</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>NA</td><td>0</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 2\n",
       "\\begin{tabular}{r|ll}\n",
       "  & Cabin & HasCabin\\\\\n",
       "  & <chr> & <dbl>\\\\\n",
       "\\hline\n",
       "\t1 & NA & 0\\\\\n",
       "\t2 & NA & 0\\\\\n",
       "\t3 & NA & 0\\\\\n",
       "\t4 & NA & 0\\\\\n",
       "\t5 & NA & 0\\\\\n",
       "\t6 & NA & 0\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 2\n",
       "\n",
       "| <!--/--> | Cabin &lt;chr&gt; | HasCabin &lt;dbl&gt; |\n",
       "|---|---|---|\n",
       "| 1 | NA | 0 |\n",
       "| 2 | NA | 0 |\n",
       "| 3 | NA | 0 |\n",
       "| 4 | NA | 0 |\n",
       "| 5 | NA | 0 |\n",
       "| 6 | NA | 0 |\n",
       "\n"
      ],
      "text/plain": [
       "  Cabin HasCabin\n",
       "1 NA    0       \n",
       "2 NA    0       \n",
       "3 NA    0       \n",
       "4 NA    0       \n",
       "5 NA    0       \n",
       "6 NA    0       "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "0"
      ],
      "text/latex": [
       "0"
      ],
      "text/markdown": [
       "0"
      ],
      "text/plain": [
       "[1] 0"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "0"
      ],
      "text/latex": [
       "0"
      ],
      "text/markdown": [
       "0"
      ],
      "text/plain": [
       "[1] 0"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create HasCabin feature\n",
    "# any_na(train$Cabin) # returns FALSE\n",
    "# describe(train$Cabin) # 687 missing - need to replace empty string values\n",
    "\n",
    "# Convert empty strings to NA in Cabin\n",
    "train$Cabin[train$Cabin == \"\"] <- NA\n",
    "test$Cabin[test$Cabin == \"\"] <- NA\n",
    "\n",
    "# n_miss(train$Cabin)\n",
    "# n_miss(test$Cabin)\n",
    "\n",
    "# Encode the HasCabin variable:\n",
    "train$HasCabin <- ifelse(!is.na(train$Cabin), 1, 0)\n",
    "test$HasCabin <- ifelse(!is.na(test$Cabin), 1, 0)\n",
    "\n",
    "# describe(train$HasCabin) # - perfect\n",
    "head(train[, c(\"Cabin\", \"HasCabin\")])  #looks good\n",
    "head(test[, c(\"Cabin\", \"HasCabin\")]) \n",
    "\n",
    "n_miss(train$HasCabin)\n",
    "n_miss(test$HasCabin)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c6d9c819",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:57.021713Z",
     "iopub.status.busy": "2025-01-18T20:50:57.020162Z",
     "iopub.status.idle": "2025-01-18T20:50:57.049968Z",
     "shell.execute_reply": "2025-01-18T20:50:57.048069Z"
    },
    "papermill": {
     "duration": 0.04165,
     "end_time": "2025-01-18T20:50:57.052472",
     "exception": false,
     "start_time": "2025-01-18T20:50:57.010822",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>\n",
       ".list-inline {list-style: none; margin:0; padding: 0}\n",
       ".list-inline>li {display: inline-block}\n",
       ".list-inline>li:not(:last-child)::after {content: \"\\00b7\"; padding: 0 .5ex}\n",
       "</style>\n",
       "<ol class=list-inline><li>2</li><li>2</li><li>1</li><li>2</li><li>1</li><li>1</li></ol>\n"
      ],
      "text/latex": [
       "\\begin{enumerate*}\n",
       "\\item 2\n",
       "\\item 2\n",
       "\\item 1\n",
       "\\item 2\n",
       "\\item 1\n",
       "\\item 1\n",
       "\\end{enumerate*}\n"
      ],
      "text/markdown": [
       "1. 2\n",
       "2. 2\n",
       "3. 1\n",
       "4. 2\n",
       "5. 1\n",
       "6. 1\n",
       "\n",
       "\n"
      ],
      "text/plain": [
       "[1] 2 2 1 2 1 1"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       ".list-inline {list-style: none; margin:0; padding: 0}\n",
       ".list-inline>li {display: inline-block}\n",
       ".list-inline>li:not(:last-child)::after {content: \"\\00b7\"; padding: 0 .5ex}\n",
       "</style>\n",
       "<ol class=list-inline><li>1</li><li>2</li><li>1</li><li>1</li><li>3</li><li>1</li></ol>\n"
      ],
      "text/latex": [
       "\\begin{enumerate*}\n",
       "\\item 1\n",
       "\\item 2\n",
       "\\item 1\n",
       "\\item 1\n",
       "\\item 3\n",
       "\\item 1\n",
       "\\end{enumerate*}\n"
      ],
      "text/markdown": [
       "1. 1\n",
       "2. 2\n",
       "3. 1\n",
       "4. 1\n",
       "5. 3\n",
       "6. 1\n",
       "\n",
       "\n"
      ],
      "text/plain": [
       "[1] 1 2 1 1 3 1"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create the FamilySize feature\n",
    "train$FamilySize <- train$SibSp + train$Parch + 1\n",
    "test$FamilySize <- test$SibSp + test$Parch + 1\n",
    "\n",
    "# Inspect the new feature\n",
    "head(train[, \"FamilySize\"])\n",
    "head(test[, \"FamilySize\"])\n",
    "\n",
    "# describe(train)\n",
    "# describe(test)\n",
    "#--test still has 1 missing fare - impute with the median\n",
    "test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)\n",
    "# describe(test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8355a02e",
   "metadata": {
    "papermill": {
     "duration": 0.007563,
     "end_time": "2025-01-18T20:50:57.068302",
     "exception": false,
     "start_time": "2025-01-18T20:50:57.060739",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Random Forest Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9effbcfa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:50:57.087028Z",
     "iopub.status.busy": "2025-01-18T20:50:57.085335Z",
     "iopub.status.idle": "2025-01-18T20:51:24.445714Z",
     "shell.execute_reply": "2025-01-18T20:51:24.443490Z"
    },
    "papermill": {
     "duration": 27.372302,
     "end_time": "2025-01-18T20:51:24.447989",
     "exception": false,
     "start_time": "2025-01-18T20:50:57.075687",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in train.default(x, y, weights = w, ...):\n",
      "“You are trying to do regression and your outcome only has two possible values Are you trying to do classification? If so, use a 2 level factor as your outcome column.”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning message in randomForest.default(x, y, mtry = param$mtry, ...):\n",
      "“The response has five or fewer unique values.  Are you sure you want to do regression?”\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest \n",
      "\n",
      "891 samples\n",
      " 11 predictor\n",
      "\n",
      "No pre-processing\n",
      "Resampling: Cross-Validated (5 fold) \n",
      "Summary of sample sizes: 713, 713, 713, 712, 713 \n",
      "Resampling results across tuning parameters:\n",
      "\n",
      "  mtry  RMSE       Rsquared   MAE      \n",
      "   2    0.3647472  0.4557431  0.2949507\n",
      "   4    0.3612627  0.4532478  0.2558578\n",
      "   7    0.3653994  0.4443990  0.2456134\n",
      "   9    0.3678295  0.4395726  0.2434292\n",
      "  12    0.3706840  0.4323223  0.2431312\n",
      "\n",
      "RMSE was used to select the optimal model using the smallest value.\n",
      "The final value used for the model was mtry = 4.\n"
     ]
    }
   ],
   "source": [
    "# Data preprocessing is now complete and we are ready to model \n",
    "# the `Survival` variable for the `test` dataset!\n",
    "\n",
    "# Train the random forest model\n",
    "rf_cv_control <- trainControl(method = \"cv\", number = 5)\n",
    "set.seed(666)\n",
    "rf_model <- train(\n",
    "  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS + HasCabin + FamilySize, \n",
    "  data = train,\n",
    "  method = \"rf\",\n",
    "  trControl = rf_cv_control,\n",
    "  tuneLength = 5\n",
    ")\n",
    "\n",
    "# Print the cross-validation results\n",
    "print(rf_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4baba7c3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:51:24.469816Z",
     "iopub.status.busy": "2025-01-18T20:51:24.468280Z",
     "iopub.status.idle": "2025-01-18T20:51:24.768484Z",
     "shell.execute_reply": "2025-01-18T20:51:24.766794Z"
    },
    "papermill": {
     "duration": 0.313628,
     "end_time": "2025-01-18T20:51:24.770936",
     "exception": false,
     "start_time": "2025-01-18T20:51:24.457308",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "test$Survived \n",
       "       n  missing distinct     Info     Mean      Gmd      .05      .10 \n",
       "     418        0      381        1   0.3999   0.3702  0.04944  0.05810 \n",
       "     .25      .50      .75      .90      .95 \n",
       " 0.09342  0.27758  0.76080  0.93266  0.96580 \n",
       "\n",
       "lowest : 0.0162999 0.0269283 0.0281149 0.0289547 0.0319296\n",
       "highest: 0.989964  0.991058  0.992763  0.995373  0.997956 "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "\n",
       "  0   1 \n",
       "272 146 "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Use the trained model to predict Survived in the test dataset\n",
    "test$Survived <- predict(rf_model, newdata = test)\n",
    "\n",
    "describe(test$Survived)\n",
    "\n",
    "# Round values in test$Survived to 0 or 1\n",
    "test$Survived <- ifelse(test$Survived >= 0.5, 1, 0)\n",
    "\n",
    "# Check the updated values\n",
    "table(test$Survived)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "402ce9c8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-01-18T20:51:24.793118Z",
     "iopub.status.busy": "2025-01-18T20:51:24.791578Z",
     "iopub.status.idle": "2025-01-18T20:51:24.822414Z",
     "shell.execute_reply": "2025-01-18T20:51:24.820699Z"
    },
    "papermill": {
     "duration": 0.044527,
     "end_time": "2025-01-18T20:51:24.824797",
     "exception": false,
     "start_time": "2025-01-18T20:51:24.780270",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"dataframe\">\n",
       "<caption>A data.frame: 6 × 2</caption>\n",
       "<thead>\n",
       "\t<tr><th></th><th scope=col>PassengerId</th><th scope=col>Survived</th></tr>\n",
       "\t<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>\n",
       "</thead>\n",
       "<tbody>\n",
       "\t<tr><th scope=row>1</th><td>892</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>2</th><td>893</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>3</th><td>894</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>4</th><td>895</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>5</th><td>896</td><td>0</td></tr>\n",
       "\t<tr><th scope=row>6</th><td>897</td><td>0</td></tr>\n",
       "</tbody>\n",
       "</table>\n"
      ],
      "text/latex": [
       "A data.frame: 6 × 2\n",
       "\\begin{tabular}{r|ll}\n",
       "  & PassengerId & Survived\\\\\n",
       "  & <int> & <dbl>\\\\\n",
       "\\hline\n",
       "\t1 & 892 & 0\\\\\n",
       "\t2 & 893 & 0\\\\\n",
       "\t3 & 894 & 0\\\\\n",
       "\t4 & 895 & 0\\\\\n",
       "\t5 & 896 & 0\\\\\n",
       "\t6 & 897 & 0\\\\\n",
       "\\end{tabular}\n"
      ],
      "text/markdown": [
       "\n",
       "A data.frame: 6 × 2\n",
       "\n",
       "| <!--/--> | PassengerId &lt;int&gt; | Survived &lt;dbl&gt; |\n",
       "|---|---|---|\n",
       "| 1 | 892 | 0 |\n",
       "| 2 | 893 | 0 |\n",
       "| 3 | 894 | 0 |\n",
       "| 4 | 895 | 0 |\n",
       "| 5 | 896 | 0 |\n",
       "| 6 | 897 | 0 |\n",
       "\n"
      ],
      "text/plain": [
       "  PassengerId Survived\n",
       "1 892         0       \n",
       "2 893         0       \n",
       "3 894         0       \n",
       "4 895         0       \n",
       "5 896         0       \n",
       "6 897         0       "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Save the updated test dataset with predictions\n",
    "gender_submission <- test %>% select(PassengerId, Survived)\n",
    "head(gender_submission)\n",
    "write.csv(gender_submission, \"gender_submission.csv\", row.names = FALSE)"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 26502,
     "sourceId": 3136,
     "sourceType": "competition"
    }
   ],
   "dockerImageVersionId": 30749,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "r",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.4.0"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 45.653901,
   "end_time": "2025-01-18T20:51:24.955743",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-01-18T20:50:39.301842",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
