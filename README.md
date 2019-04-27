# Predict Assessment Scores - Sourced from Kaggle
#### MSDS696 Data Science Practicum II

## PROJECT OVERVIEW
One measure of a successful online delivery course is the student’s assessment score or grade.  Analyzing what features have the most impact on assessment scores is a great way to identify potential areas of improvement to an online learning experience.  The Open University Learning Analytics dataset sourced from Kaggle contains student demographics,  click data on course materials, student registrations, and assessment score information.  I utilized the available data points, along with feature engineering, to train supervised learning models to predict assessment outcomes.

For this project, I spent most of my time training regression models to predict assessment scores.  It was an amazing exercise in learning but I was not entirely happy with the results.  For that reason, I also added some classification models to this research and simply predicted if the student passed or failed.  Of course, the results where better as I was choosing pass or fail as opposed to a score from 0 to 100.  In the process of seeking better results, I feel like I gained some confidence in understanding the complexities of machine learning.  

## DATA

The data and variable descriptions came from from **Kaggle:**
https://www.kaggle.com/rocki37/open-university-learning-analytics-dataset/activity & https://analyse.kmi.open.ac.uk/open_dataset

### Open University Datasets

**Courses** - Modules and there presentations,
22 rows, 3 variables

| Variable       | Description          |
|:------------- |:-------------|
| code_module | Code name of the module. |
| code_presentation | Code name of the presentation. |
| length | Length in days of module/presentation. |

**Assessments** - Assessments in module presentations, 206 rows, 6 variables

| Variable       | Description          |
|:------------- |:-------------|
| code_module | ID code of module. |
| code_presentation | ID code of presentation. |
| id_assessment | ID number of assessment. |
| assessment_type | Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA) and Final Exam (Exam). |
| date | Number of days since the start of the module-presentation. |
| weight | % weight of assessment. |

**Virtual Learning Environment (VLE)** - Available materials in the virtual learning environment (VLE),
6364 rows, 6 variables

| Variable       | Description          |
|:------------- |:-------------|
| id_site | ID number of material. |
| code_module | ID code of module. |
| code_presentation | ID code of presentation. |
| activity_type | Role associated with material. |
| week_from | Planned use from week. |
| week_to | Planned use to week. |

**Student Info** - Demographic information about students and their final result,
32,593 rows, 12 variables

| Variable       | Description          |
|:------------- |:-------------|
| code_module | ID code of module. |
| code_presentation | ID code of presentation. |
| id_student | ID number of student. |
| gender | Students gender. |
| region | Geographic region. |
| highest_education | highest student education level. |
| imd_band |  Index of Multiple Depravation. |
| age_band | Students age. |
| num_of_prev_attempts | Number of times module attempted by student. |
| studied_credits | Number of credits for module. |
| disability | Indicates if student declared disability. |
| final_result | Students final result of module. |

**Student Registration** - Student registration information for the module presentation,
32,593 rows, 5 variables

| Variable       | Description          |
|:------------- |:-------------|
| code_module | ID code of module. |
| code_presentation | ID code of presentation. |
| id_student | ID number of student. |
| date_registration | Date of students registration in days relative to modules start. |
| date_unregistration | Date of students registration in days relative to modules start.  Blank for students who complete course.  Students that unregister have a final result of 'Withdrawn'. |

**Student Assessment** - Results of student assessments, 173,912 rows, 5 variables

| Variable       | Description          |
|:------------- |:-------------|
| id_assessment | ID number of assessment.
| id_student | ID number of student. ||
| date_submitted | Date of students submission in days relative to modules start. |
| is_banked | Flag indicating assessment result transferred from previous presentation. |
| score | Assessment score from 0 - 100.  40 is considered 'Fail'. |

**Student VLE** - Students interactions with materials in the VLE,
1,0655,280 rows, 6 variables

| Variable       | Description          |
|:------------- |:-------------|
| code_module | ID code of module. |
| code_presentation | ID code of presentation. |
| id_student | ID number of student. |
| id_site | ID number of material. |
| date | Date of student interaction with material. Measured in number of days since module start. |
| sum_click | Number of interactions by student with material in that day. |

## JOINS
The first challenge was to bring all of this data together into a cohesive dataset that could be used for machine learning.  I knew I had 173,912 assessments with scores to train my models.  It quickly became apparent their were multiple assessments for each student.  After my joins, a student could have multiple assessment rows with many common features and a different target score.  This could make training a model problematic.   

![alt text](images/join_1.png "join")

In an effort to gain a better understanding of machine learning, I decided to compare the results of two versions of the dataset in the regression models.  The first dataset is grouped by the student, includes the material click data, and the average assessment score for each student is the target.  The second dataset does not include the click data and has a row for each assessment score for each student as the target.  Some features in the second dataset are not categorical and may provide enough variance to make decent predictions.  

**Click Dataset**
<table align="center">
    <tr>
        <td align="center"><b>Description</b></td>
        <td align="left">Average score per student and total clicks broken out by  course activity type.</td>
    </tr>
    <tr>
        <td align="center"><b>Structure</b></td>
        <td align="left">more features, less rows</td>
    </tr>
    <tr>
        <td align="center"><b>Rows/Features</b></td>
        <td align="left">37,030 rows, 38 features</td>
    </tr>
</table>


**Assessment Dataset**
<table align="center">
    <tr>
        <td align="center"><b>Description</b></td>
        <td align="left">All assessments with many repeated features (many assessments per student) and different outcome score.</td>
    </tr>
    <tr>
        <td align="center"><b>Structure</b></td>
        <td align="left">less features, more rows</td>
    </tr>
    <tr>
        <td align="center"><b>Rows/Features</b></td>
        <td align="left">153,537 rows, 19 features</td>
    </tr>
</table>

For both datasets, records with a final status of 'Witdrawn' and records that were missing an imd band were removed.

## FEATURE ENGINEERING

The following variables were feature engineered:

| Variable       | Description    |Dataset|
|:------------- |:-------------|:-------------|
| Average Click | Average clicks across all activity types for a student. |click |
| Access Date | date minus date submitted for both datasets. Date is the final date to complete assessment and date submitted is when it was taken. |click and assessment |
| Module length to number of credits ratio | length of module presentation in days divided by the number of credits for the module. |click and assessment |
| Total clicks by activity type by student  | Total clicks on course materials for each student (e.g.page, questionnaire, quiz). |click |


## EDA (EXPLORATORY DATA ANALYSIS)

### Correlation Matrix
 I created a correlation matrix for both datasets and did not see very strong correlations between any predictor variables.  I therefore did not remove any predictor variables.  

 <table>
   <tbody>
     <tr>
       <th align="center">Click Dataset</th>
       <th align="center">Assessment Dataset</th>
     </tr>
     <tr>
       <td><img src="images/corr_click.png"></td>
       <td><img src="images/corr_assess.png"></td>
     </tr>
  </table>

### Bar Plots
 Many bar plots were utilized to better understand the data. Below are a few that I found the most interesting.

 <table>
   <tbody>
     <tr>
       <th align="center">Assessment Type</th>
       <th align="center">Age Band</th>
       <th align="center">Final Result</th>
     </tr>
     <tr>
       <td><img src="images/assess_type_bar.png"></td>
       <td><img src="images/age_bar.png"></td>
       <td><img src="images/final_res_bar.png"></td>
     </tr>
  </table>

Tutor marked assessments (TMA) are the highest proportion of assessments types and final exams the lowest. Almost 70% of the students are less than 35 years old and over 20% of the students had a final result of 'Failed' with the remainder passing or passing with distinction.

### Boxplots
 <table>
   <tbody>
     <tr>
       <th align="center">Assessment Type</th>
       <th align="center">Age Band</th>
       <th align="center">Course Module</th>
     </tr>
     <tr>
       <td><img src="images/assess_type_box.png"></td>
       <td><img src="images/age_box.png"></td>
       <td><img src="images/course_box.png"></td>
     </tr>
  </table>

Some of the more interesting boxplots compared score to assessment type, age band, and course module.  Computer marked assessments had the highest median and exams the lowest.  When comparing score to age band, the age band with the highest score median was 55 and over.  Looking into the course setup, methods of delivery, and assessment details of course modules with the highest score median may help build more successful courses. Course modules EEE, FFF, and BBB had the highest score medians.  Using median for this part of the analysis seems to be a good measure as the median is less impacted by outliers.   

### Distribution of Data

I experimented with many scatterplots and other visualizations to get the basic shape of the data.  The better results came in a scatterplot that showed the correlation between average clicks (feature engineered predictor) and score.  

**Average Clicks/Score Scatter**

![alt text](images/avg_clicks_scatter.png "scatter avg clicks score")

It's very clear that the lower scores had lower average clicks.  To be fair there are plenty of higher scores with lower average clicks as well.  There does, however, seem to be a slight positive correlation.  Intuitively, this makes sense, as  more time spent working with course materials should result in a better course score.  

**Score Histograms**

<table>
  <tbody>
    <tr>
      <th align="center">Click Dataset</th>
      <th align="center">Assessment Dataset</th>
    </tr>
    <tr>
      <td><img src="images/hist_click.png"></td>
      <td><img src="images/hist_assess.png"></td>
    </tr>
 </table>

 Both datasets are left skewed, although the assessment dataset seems to be more left skewed than the click dataset.  The assessment dataset has 2 large spikes around 80 and 100.  The click dataset seems to be the closest of the two to a normal distribution.  Looking at this it does seem like linear models may not perform well with this data.  In the analysis section we'll see this does prove true as non-linear models have the best results.  

##  ANALYSIS

### Model Result Measures

I chose to measure the model results with the following 2 metrics:

**Adjusted r2** - r2 is the percentage of the variation in response variables that is explained by the model. (ref r2)  Adjusted r2 adjusts for multiple predictors and only increases the score if a predictor improves a model more than what chance would predict(ref adjusted r2).  The higher the adjusted r2, the better.

**Root Mean Square Error (RMSE)** - RMSE is the "standard deviation of the residuals" (ref RMSE).  The closer to zero (exact prediction), the better.  

###  GridSearchCV

I utilized GridSearchCV to perform 5 fold cross validation and identify the optimal parameters for all of my models.  It did take a long time to run each model, but seemed to worth it to get the best parameters settings.

**Sample of GridSearchCV Code with parameter grid**  

``` python

rf = RandomForestRegressor(random_state=42)

parameters = { 'max_features':np.arange(5,10),'n_estimators':[500],'min_samples_leaf': [10,50,100,200,500]}

rf_gs = GridSearchCV(rf,parameters,scoring=scoring,refit='r2',cv=5,return_train_score=True)

start_time = timer(None)
rf_gs.fit(X_train, y_train)
timer(start_time)
```
###  Supervised Learning - Regression

Write something here about linear regression and regularization. score.

**Click Dataset - Top 3 Models**
<table align="center">
    <tr>
        <td align="center"><b>Model</b></td>
        <td align="center"><b>Adjusted r2</b></td>
        <td align="center"><b>RMSE</b></td>
    </tr>
    <tr>
        <td align="left">Random Forest Regressor</td>
        <td align="left">0.5049</td>
        <td align="left">11.4554</td>
    </tr>
    <tr>
        <td align="left">Neural Netr</td>
        <td align="left">0.4628</td>
        <td align="left">11.9323</td>
    </tr>
    <tr>
        <td align="left">XGBoost</td>
        <td align="left">0.5278</td>
        <td align="left">11.1868</td>
    </tr>
</table>

![alt text](images/click_results_scat.png "click_results_scat")

**Assessment Dataset - Top 3 Models**
<table align="center">
    <tr>
        <td align="center"><b>Model</b></td>
        <td align="center"><b>Adjusted r2</b></td>
        <td align="center"><b>RMSE</b></td>
    </tr>
    <tr>
        <td align="left">Random Forest Regressor</td>
        <td align="left">0.3748</td>
        <td align="left">14.4142</td>
    </tr>
    <tr>
        <td align="left">Neural Netr</td>
        <td align="left">0.3308</td>
        <td align="left">14.9135</td>
    </tr>
    <tr>
        <td align="left">XGBoost</td>
        <td align="left">0.3455</td>
        <td align="left">14.7488</td>
    </tr>
</table>

![alt text](images/assess_results_scat.png "assess_results_scat")

###  Random Forest - Important Features

![alt text](images/rf_imp.png "rf_imp")

**Important Features Removed**

<table align="center">
    <tr>
        <td align="center"><b>Model</b></td>
        <td align="center"><b>Adjusted r2</b></td>
        <td align="center"><b>RMSE</b></td>
    </tr>
    <tr>
        <td align="left">Random Forest Regressor</td>
        <td align="left">0.5250</td>
        <td align="left">11.2207</td>
    </tr>
</table>

###  Supervised Learning - Classification

<table align="center">
    <tr>
        <td align="center"><b>Model</b></td>
        <td align="center"><b>Kappa r2</b></td>
        <td align="center"><b>AUC</b></td>
        <td align="center"><b>Precision</b></td>
        <td align="center"><b>Recall</b></td>
    </tr>
    <tr>
        <td align="left">Logistic Regression</td>
        <td align="left">0.4750</td>
        <td align="left">0.85567</td>
        <td align="left">0.85</td>
        <td align="left">0.94</td>
    </tr>
    <tr>
        <td align="left">Random Forest Classifier</td>
        <td align="left">0.6020</td>
        <td align="left">0.91729</td>
        <td align="left">0.87</td>
        <td align="left">0.97</td>
    </tr>
</table>

## CONCLUSIONS

* Regression Models didn’t get the best results, more features may improve results


* Simple Word Counts Impressive

* GridSearchCV worked great to identify optimal parameters - Further hypertuning may improve results while being cautious about overfitting

* Predicting assessment scores would have been easier if there was one score per student, made things complex

* Ensemble models are powerful and produce impressive results

* It makes sense that classification performed better than regression, as classification had a 50% chance of getting it right


## REFERENCES

**Youtube Presentation**
https://youtu.be/0BFaGmPbY_k

George, P. (2018, December 13). Working at Tesla Means Being in an 'Abusive Relationship' With Elon Musk: Report. Retrieved from https://jalopnik.com/working-at-tesla-means-being-in-an-abusive-relationship-1831072258

Schneider, M. (2017, July 26). Google Gets 2 Million Applications a Year. To Have a Shot, Your Resume Must Pass the '6-Second Test'. Retrieved from https://www.inc.com/michael-schneider/its-harder-to-get-into-google-than-harvard.html

Thibodeaux, W. (2018, September 19). 67 Percent of Recruiters Say It's Harder Than Ever to Find Talent. Here's How to Beat the Odds. Retrieved from https://www.inc.com/wanda-thibodeaux/67-percent-of-recruiters-say-its-harder-than-ever-to-find-talent-heres-how-to-beat-odds.html
