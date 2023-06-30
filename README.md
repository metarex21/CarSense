# CarSense: Visualizing Indian Car Data
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and analysis for the CarSense project, focused on visualizing and analyzing Indian car data. In this project, we perform exploratory data analysis (EDA) on a dataset of Indian cars to gain insights and visualize various aspects of the data.

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/bhanupratapbiswas/indian-cars-data-analysis-and-visualization). It contains information about different car models including their make, variant, ex-showroom price, displacement, cylinders, drivetrain, and various other features.

## Project Structure

- `notebooks/`: This directory contains Jupyter notebooks with the code for the EDA analysis, data visualization, and correlation analysis.
- `dataset/`: This directory contains the dataset file `cars_ds_final.csv`.

## Analysis Steps

1. **Data Loading and Exploration**: We start by loading the dataset and exploring its structure, including column names, data types, and basic statistics.

2. **Data Visualization**: We visualize the data using various plots and charts to understand the distribution of numerical columns, count of unique car makes, and relationships between variables. We analyze trends, patterns, and correlations within the dataset.

3. **Correlation Analysis**: We perform correlation analysis to identify relationships between numerical variables and the target variable (car make). This helps us understand the influence of different features on the car make.

4. **Grouping and Aggregation**: We group the data based on specific criteria and perform aggregations to derive meaningful insights. This includes calculating average prices by manufacturer, counting the number of unique car models in price ranges, and more.

5. **Conclusion**: We summarize the key findings and insights obtained from the analysis. This helps in understanding the dataset and can be used as a foundation for further analysis or machine learning tasks.

## Getting Started

To run the code and perform the analysis, follow these steps:

1. Clone the repository: `git clone https://github.com/metarex21/CarSense.git`
2. Navigate to the project directory: `cd CarSense`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Open the Jupyter notebooks in the `notebooks/` directory to explore the analysis code.

## Contribution

Contributions to this project are welcome! If you have any suggestions, improvements, or additional analysis ideas, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
