# American-Express-1D-second-look-lending


### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                                     |
|------------------|---------------|--------------------------------------------------------------------------        |
| Melinda Li       | @melindaahli  | Data aggregation, LightGBM Model Development, Frontend                           |
| Grace Chiu       | @grchiu6      | Exploratory data analysis (EDA), LightGBM Model Development                      |
| Arhaan Parikh    | @parikh24     | Exploratory data analysis (EDA), Logistic Regression Model Development, Frontend |
| Atai Zheenbekov  | @ataizh       | Feature Engineering, Normalizing Data, Random Forest Model Development           |
| Rachel Dayts     | @racheldayts  | Data Cleaning, Logistic Regression Model Development, Documentation              |
| Steve Yang       | @syang1290    | Normalizing Data, Random Forest Model Development, Frontend                      |
| Shuzana Rahman   | @Shuzana2405  | Feature Engineering, LightGBM Model Development                                  |

---

## 🎯 **Project Highlights**

- Developed a two-stage machine learning lending system using Logistic Regression, Random Forest, and LightGBM to identify creditworthy applicants overlooked by traditional lending approaches.
- Processed and analyzed over 1 million loan applications, incorporating feature engineering, hyperparameter optimization, and probability calibration to improve default-risk assessment.
- Designed an interactive frontend to demonstrate the lending decision process, visualize predicted default probabilities, and communicate personalized lending outcomes.
- Implemented risk-based decision policies and second-look evaluation strategies to balance financial risk, profitability, and equitable access to credit.

  Note: Update with resulting key metrics when full project finished
---

## 👩🏽‍💻 **Setup and Installation**

**Clone the repository by running in terminal:** 

git clone https://github.com/Break-Through-Tech/American-Express-1D-second-look-lending.git

**Install dependencies by running in terminal:** 

pip install -r requirements.txt

**How to access the datasets:**

1. Go into Challenge-Project-Overview.md
2. Go to drive link with data
3. Download zip file and unzip it
4. Move files into project directory, making sure to only have folders "test" and "train" inside the data folder

**How to run the notebook or scripts:**

Run python run_pipeline.py to preprocess the data...

Note: Update with further instructions when full project is completed

---

## 🏗️ **Project Overview**

As part of the Break Through Tech AI Program, our team partnered with American Express to develop a two-stage machine learning lending system designed to identify creditworthy applicants who may be overlooked by traditional lending approaches, particularly those with limited credit histories (thin-file applicants). Our approach utilizes Logistic Regression for initial default-risk assessment, followed by a second-look evaluation using Random Forest or LightGBM. The project encompasses data preprocessing, exploratory data analysis, feature engineering, model optimization, probability calibration, business policy evaluation, and the development of an interactive frontend to demonstrate the lending decision process. By leveraging machine learning to identify additional lending opportunities, our project aims to expand access to credit while helping financial institutions balance profitability, risk management, and financial inclusion.

---

## 📊 **Data Exploration**

- **Dataset:** Synthetic consumer lending data provided by American Express, containing approximately 1.2 million loan applications in CSV format
- **Data Characteristics:** Numerical and categorical features representing applicant financial profiles, credit histories, borrowing behavior, and loan repayment outcomes.
- **Exploratory Data Analysis (EDA):** Examined feature distributions, missing values, correlations, default rates, and differences between applicant segments.
- **Key Insights:** Approximately 30% of applicants were classified as thin-file, highlighting the importance of evaluating model performance and lending decisions across different credit history groups.
- **Challenges:** Addressed incomplete credit histories, class imbalance, and the trade-offs between predictive performance, lending profitability, and financial inclusion.

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

Note: Will fill in once full project finished

---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

Note: Will fill in once full project finished

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

Note: Will fill in once full project finished

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

Note: Need to ask CA

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

Note: Will fill in once full project finished

---

## 🙏 **Acknowledgements**

We would like to thank our Challenge Advisor, AI Studio Coach, and American Express and Break Through Tech representatives for their support and guidance on our project.
