# dga-explainability
Code and Result repository for the paper "SHAP Interpretations of Tree and Neural Network DNS Classifiers for Analyzing DGA Family Characteristics" under review by IEEE Access.  
  
Implementation folder: Code for reproducing our experiments organized as follows:   
- filter_tranco_names: Code to obtain Tranco List and filter it to exclude DGA names. DGA names are provided by the DGArchive repository.  
- reduction_and_labeling.py: Takes as input names from Tranco list and DGArchive. Well-known valid DNS suffixes are then removed from these names.  
- feature_extraction: Extracts features from given domain names.  
- hyperparameters: Hyperparameter optimization considering multiple models.  
- explanations: Program to train/test the machine learning models and provide explanations.    
  
Results folder: Organized per DGA family (one folder includes plots considering instances from all DGA families). Each family has:  
- summary plots: SHAP summary plots considering XGBoost and MLP classifiers (xgboost and mlp respectively). Plots tagged as "original" are those returned by the shap package, whereas plots tagged as "xlim-11" are those returned by the shap package with the horizontal axis of SHAP values scaled between -1 and +1  
- dependence plots: SHAP dependence plots for multiple features (Reputation, Length, etc.) and XGBoost/MLP classifiers.  
- force plots: Multiple SHAP force plots for specific instannces considering XGBoost and MLP classifiers (xgboost, mlp respectively). Instances names are included within the name of the file (in order not to cause confusion with the dot delimeter, dots are substituted by "+" in the name of the file). Prediction is the output of the XGBoost and MLP classifier, i.e. 0/1 for XGBoost and [0, 1] for MLP.  
