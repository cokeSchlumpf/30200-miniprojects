# Miniprojekt: Classification Competition

Von: Till Staab & Michael Wellner

# Experimente mit R

Die beiden Datasets aus Competition 1 und 2 (über caret) haben wir genutzt um Modelle mit den folgenden Algorithmen in R zu trainieren (siehe auch [exploration/SecondApproachWithCaret.R](./exploration/SecondApproachWithCaret.R):

* Gradient Boosting Model GBM (method = gbm)
* Extrem Gradient Boosting (method = xgbTree)
* Random Forest (method = rf)
* Support Vector Machine (method = svmLinear)

Bester Score für Competition 1: **84% mit GBM** 
Bester Score für Competition 2: **76% mit GBM** [result/data_comp_2_StaabWellner.csv](./result/data_comp_StabWellner.csv)

... jeweils basierend auf X-Validation Trainings/ Test-Sets

# Experimente Python 
 
Mit Python haben wir ebenfalls klasische Methode mit SciKit-Learn, aber auch Modelle mit Keras probiert (siehe auch [Notebook](./exploration/Untitled.ipynb)).

* SVM
* DecissionTree
* RandomForrest
* MLP
* Keras mit einem Hidden-Layer (versch. Parameter probiert)

Bester Score für Competition 1: **86% mit Keras** [results/data_comp_1_StaabWellner.csv](./result/data_comp1_StaabWellner.csv).