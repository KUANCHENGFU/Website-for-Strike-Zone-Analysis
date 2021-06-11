# Website for Strike Zone Analysis

## Overview
In this project, a framework is first proposed to classify whether a given pitch is a strike or a ball and further predict its called strike probability. After that, on the basis of the proposed framework, a website is designed and managed using Flask, Bootstrap, and AWS to visualize each umpire's strike zone shape under various conditions, such as count, batter side, pitcher side, and so on. The proposed framework is basically composed of five parts as follows:

1. Feature Preprocessing
2. Model building (LightGBM) and parameter tuning (Bayesian optimization)
3. Model evaluation (overall accuracy and confusion matrix)
4. Feature Selection (sequential forward selection)
5. Probability calibration (Platt scaling and isotonic regression)


## Link to the Project
1. https://nbviewer.jupyter.org/github/KUANCHENGFU/Website-for-Strike-Zone-Analysis/blob/main/source/Called_strike_probability.ipynb

&nbsp;

![Website Screeshot](https://github.com/KUANCHENGFU/Website-for-Strike-Zone-Analysis/blob/main/source/website%20screenshot.png)
