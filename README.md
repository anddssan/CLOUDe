# CLOUDe
CLOUDe is an R implementation of Campelo dos Santos, DeGiorgio and Assis' (2023) suite of machine learning methods for predicting evolutionary targets of gene deletion events from expression data in two species.

-------------
Citing CLOUDe
-------------
Thank you for using the CLOUDe classifier and predictor.

If you use this R package, then please cite:
    AL Campelo dos Santos, M DeGiorgio, R Assis. In Preparation

----------------
Reporting issues
----------------
If you find any issues running CLOUDe, then please contact Andre Luiz Campelo dos Santos directly through acampelodossanto@fau.edu.
	
---------------	
Getting started
---------------
Before you are able to use the CLOUDe R package, you will need to have the dplyr, mvtnorm, tensorflow, keras, ranger, devtools, liquidSVM, and xgboost libraries installed in your R environment. These libraries can be installed with the following commands in R:

    install.packages("dplyr")
    install.packages("mvtnorm")
    install.packages("tensorflow")
    install.packages("keras")
    install.packages("ranger")
    install.packages("devtools")
    library(devtools)
    install_version("liquidSVM", "1.2.0")
    install.packages("xgboost")
  
The CLOUDe package comes with the script CLOUDe.R and an ExampleFiles directory containing example files to help you get started.
  
The CLOUDe package can be loaded in R with the command:

    source("CLOUDe.R")

Note that this command assumes that you are in the directory where the CLOUDe.R script is located.

To run CLOUDe, you will need to provide an input file containing expression data for sets of genes that underwent deletion events. The format of this file is described in the next section, and an example file called EmpiricalData.data is provided in the ExampleFiles directory.

--------------------
Format of input file
--------------------
A space-delimited file with N+1 rows and 3m columns, where N is the number of genes and m is the number of conditions (e.g., tissues, ages, etc.) for expression data. The first row is a header.

Each row (rows 2 to N+1) is a deletion event, and each column is an absolute log10(x+1) expression value at a particular condition in a particular gene corresponding to the deletion event. The first m columns are the expression values for the m conditions in the derived gene, the next m columns are the expression values for the m conditions in the survived gene, and the last m columns are the expression values for the m conditions in the lost gene. Specifically, columns j, m+j, and 2m+j represent the expression values at condition j (j = 1, 2, ..., m) in the derived, survived, and lost genes, respectively.

The header takes the form:

    "eD1" "eD2" ... "eDm" "eS1" "eS2" ... "eSm" "eL1" "eL2" ... "eLm"

where eD1 to eDm denote derived gene expression values for conditions 1 to m, eS1 to eSm denote survived gene expression values for conditions 1 to m, and eL1 to eLm denote lost gene expression values for conditions 1 to m.

The ExampleFiles directory contains a file called EmpiricalData.data, which illustrates this format for N=100 deletion events at m=6 tissues.

-------------------------------------------
Generating training data from an OU process
-------------------------------------------
Training data can be generated with the command:

    GenerateTrainingData(m, Nk, training_prefix, logThetaMin, logThetaMax, logAlphaMin, logAlphaMax, logSigmaSqMin, logSigmaSqMax, empiricalFile)

where m is the number of conditions, Nk is the number of training observations for each of the two classes, training_prefix is the prefix given to all files output by this function, empiricalFile is the full name of the input file formatted as described in "Format of input file", and the rest of the parameters are used to define the ranges that the evolutionary parameters theta (optimal expression), alpha (strength of selection) and sigmaSq (strength of phenotypic drift) of the OU process are drawn from.

The GenerateTrainingData() function also outputs the p=3m features to the file training_prefix.features, a one-hot encoded matrix of classifications for all training observations to the file training_prefix.classes, a matrix of predicted expression optima for all training observations to the file training_prefix.responses, raw simulated data to the file training_prefix.data, and the means and standard deviations for each of the p features and each of the 3m model parameters in training_prefix.X_stdparams and training_prefix.Y_stdparams, respectively.

---------------
Training CLOUDe
---------------
Due to their speed, training of the CLOUDe support vector machine and random forest are performed on-the-fly while being applied to test data. Thus, this step only needs to be performed if using the CLOUDe neural network or extreme gradient boosting. Otherwise, the user can skip this section. 

The CLOUDe neural network and extreme gradient boosting models are trained on the training data outputted by the GenerateTrainingData() function, as described in "Generating training data from an OU process". However, because estimating their many hyperparameters is a time-consuming process, we implement hyperparameter tuning of the CLOUDe neural network and extreme gradient boosting predictors separately from final model training. For the CLOUDe neural network, we perform five-fold cross-validation to identify optimal hyperparameters for the predictor with num_layer layers (num_layer in {0, 1, 2, 3, 4, 5}), regularization tuning parameter lambda, and elastic net tuning parameter gamma. Conditional on the optimal lambda and gamma hyperparameters, CLOUDe then fits a neural network predictor with num_layer hidden layers. The optimal number of hidden layers is chosen as the one with the smallest validation loss. For the CLOUDe extreme gradient boosting models, we perform five-fold cross-validation to identify optimal hyperparameters for the predictor with maximum depth of trees of max_depth (max_depth in {1, 2, 3, 4, 5, 6}), regularization tuning parameter lambda, elastic net tuning parameter gamma, and learning rate parameter eta. Conditional on the optimal lambda, gamma, and eta, hyperparameters, CLOUDe then fits an extreme gradient boosting predictor with maximum depth of trees of max_depth. The optimal maximum depth of trees is chosen as the one with the smallest validation loss.

The ClassifierCVnn() and ClassifierCVxgb() functions are used to respectively train the CLOUDe neural network and extreme gradient boosting methods to predict classes ("Redundant" or "Unique"), and the PredictorCVnn() and PredictorCVxgb() functions are used to train the CLOUDe neural network and extreme gradient boosting methods to predict evolutionary parameters (theta1, theta2, and log10(sigmaSq/(2*alpha))). For CLOUDe neural network, we consider log(lambda) evenly distributed within [log_lambda_min, log_lambda_max] for num_lambda values, and gamma evenly distributed within [gamma_min, gamma_max] with num_gamma values, assuming a batch size of batchsize observations per epoch and trained for num_epochs epochs with the commands:

    ClassifierCVnn(num_layers, batchsize, num_epochs, log_lambda_min, log_lambda_max, num_lambda, gamma_min, gamma_max, num_gamma, training_prefix)
    PredictorCVnn(num_layers, batchsize, num_epochs, log_lambda_min, log_lambda_max, num_lambda, gamma_min, gamma_max, num_gamma, training_prefix)

where num_layers is the number of layers (0, 1, 2, 3, 4 or 5) of the neural network, batchsize is the number of training observations used in each epoch, num_epochs is the number of training epochs, hyperparameter lambda is drawn from log10(lambda) in interval [log_lambda_min, log_lambda_max] for num_lambda evenly spaced points, hyperparameter gamma is drawn from interval [gamma_min, gamma_max] for num_gamma evenly spaced points, and training_prefix is the prefix to all files outputted by this function.

These functions output the set of optimal hyperparameters chosen through cross-validation to the files training_prefix.nn.num_layers.classifier_cv and training_prefix.nn.num_layers.predictor_cv, and the fitted neural network models in TensorFlow format to the files training_prefix.nn.num_layers.classifier.hdf5 and training_prefix.nn.num_layers.predictor.hdf5.

For CLOUDe extreme gradient boosting, we consider eta evenly distributed within [eta_min, eta_max] with num_eta values, log(lambda) evenly distributed within [log_lambda_min, log_lambda_max] for num_lambda values, and gamma evenly distributed within [gamma_min, gamma_max] with num_gamma values with the commands:

    ClassifierCVxgb(max_depth, eta_min, eta_max, num_eta, log_lambda_min, log_lambda_max, num_lambda, gamma_min, gamma_max, num_gamma, training_prefix)
    PredictorCVxgb(max_depth, eta_min, eta_max, num_eta, log_lambda_min, log_lambda_max, num_lambda, gamma_min, gamma_max, num_gamma, training_prefix)

where max_depth is the maximum depth (1, 2, 3, 4, 5 or 6) of trees in the models, hyperparameter eta is drawn from interval [eta_min, eta_max] for num_eta evenly spaced points, hyperparameter lambda is drawn from log10(lambda) in interval [log_lambda_min, log_lambda_max] for num_lambda evenly spaced points, hyperparameter gamma is drawn from interval [gamma_min, gamma_max] for num_gamma evenly spaced points, and training_prefix is the prefix to all files outputted by this function.

These functions output the set of optimal hyperparameters chosen through cross-validation to the files training_prefix.xgb.max_depth.classifier_cv and training_prefix.xgb.max_depth.predictor_cv.

---------------------------
Performing test predictions
---------------------------
CLOUDe can predict classes ("Redundant" or "Unique") and evolutionary parameters (theta1, theta2, and log10(sigmaSq/(2*alpha))) for each gene in a test dataset. 

To predict classes, the following commands can be used depending on the chosen method:

    CLOUDeClassifyNN(training_prefix, testing_prefix, num_layers)   # neural network
    CLOUDeClassifyXGB(training_prefix, testing_prefix, max_depth)   # extreme gradient boosting
    CLOUDeClassifyRF(training_prefix, testing_prefix)               # random forest
    CLOUDeClassifySVM(training_prefix, testing_prefix)              # support vector machine
  
where training_prefix and testing_prefix are the prefixes to training and testing feature files that were outputted by the GenerateTrainingData() function, num_layers is the number of layers (0, 1, 2, 3, 4 or 5) of the neural network, and max_depth is the maximum depth of trees (1, 2, 3, 4, 5 or 6) of the extreme gradient boosting models.
  
The CLOUDeClassifyNN(), CLOUDeClassifyXGB(), CLOUDeClassifyRF(), and CLOUDeClassifySVM() functions output predicted classes and probabilities for each deletion event in the test dataset to the respective files

    test_prefix.nn.num_layers.classifications
    test_prefix.xgb.max_depth.classifications
    test_prefix.rf.classifications
    test_prefix.svm.classifications

and

    test_prefix.nn.num_layers.probabilities
    test_prefix.xgb.max_depth.probabilities
    test_prefix.rf.probabilities
    test_prefix.svm.probabilities

To predict evolutionary parameters, the following commands can be used depending on the chosen algorithm:

    CLOUDePredictNN(training_prefix, testing_prefix, num_layers)    # neural network
    CLOUDePredictXGB(training_prefix, testing_prefix, max_depth)    # extreme gradient boosting
    CLOUDePredictRF(training_prefix, testing_prefix)                # random forest
    CLOUDePredictSVM(training_prefix, testing_prefix)               # support vector machine
  
where training_prefix and testing_prefix are the prefixes to training and testing feature files that were outputted by the GenerateTrainingData() function, num_layers is the number of layers (0, 1, 2, 3, 4 or 5) of the neural network, and max_depth is the maximum depth of trees (1, 2, 3, 4, 5 or 6) of the extreme gradient boosting models.
  
The CLOUDePredictNN(), CLOUDePredictXGB(), CLOUDePredictRF(), and CLOUDePredictSVM() functions output the 3m predicted evolutionary parameters for each gene in the test dataset to the respective files 

    test_prefix.nn.num_layers.predictions
    test_prefix.xgb.max_depth.predictions
    test_prefix.rf.predictions
    test_prefix.svm.predictions

-----------------------------
Example application of CLOUDe
-----------------------------
Within the R environment, set the working directory to the directory containing both the CLOUDe.r script and the subdirectory ExampleFiles containing the example files. 

Load the functions of the CLOUDe package by typing the command:
  
    source("CLOUDe.r")

Next, generate a training dataset of 100 observations in six conditions for each of the two classes, and store the training data with prefix Training, by typing the command:

    GenerateTrainingData(6, 100, "ExampleFiles/Training", 0, 5, 0, 3, -2, 3, "ExampleFiles/EmpiricalData.data")

The above operation will output the files

    Training.data
    Training.features
    Training.X_stdparams
    Training.classes
    Training.responses
    Training.Y_stdparams

for which log10(theta) is drawn between 0 and 5, log10(alpha) is drawn between 0 and 3, and log10(sigmaSq) is drawn between -2 and 3.

To train the CLOUDe neural network with 2 hidden layers on this training dataset using five-fold cross-validation with hyperparameters log(lambda) drawn from {-3, -2, -1, 0, 1, 2, 3} and gamma drawn from {0, 0.5, 1}, assuming a batch size of 50 observations per epoch and trained for 50 epochs, type the commands:

    ClassifierCVnn(2, 50, 50, -3, 3, 7, 0, 1, 3, "ExampleFiles/Training")
    PredictorCVnn(2, 50, 50, -3, 3, 7, 0, 1, 3, "ExampleFiles/Training")

The above operations will output the files 

    Training.nn.2.classifier_cv
    Training.nn.2.classifier.hdf5
    Training.nn.2.predictor_cv
    Training.nn.2.predictor.hdf5

To train the CLOUDe extreme gradient boosting models with a maximum depth of tress of 4 on this training dataset using five-fold cross-validation with hyperparameters eta drawn from {1, 2, 3}, log(lambda) drawn from {-3, -2, -1, 0, 1, 2, 3} and gamma drawn from {0, 0.5, 1}, type the commands:

    ClassifierCVxgb(4, 1, 3, 3, -3, 3, 7, 0, 1, 3, "ExampleFiles/Training")
    PredictorCVxgb(4, 1, 3, 3, -3, 3, 7, 0, 1, 3, "ExampleFiles/Training")

The above operations will output the files 

    Training.xgb.4.classifier_cv
    Training.xgb.4.predictor_cv

Note that we perform training and hyperparameter tuning of the CLOUDe neural network and extreme gradrient boosting models in advance of application to test data, whereas the CLOUDe support vector machine and random forest classifiers and predictors are trained on-the-fly during application to test data.

Finally, to predict classes and evolutionary parameters for genes in an empirical dataset with the CLOUDe two-layer neural network, random forest, and support vector machine, type the commands:

    CLOUDeClassifyNN("ExampleFiles/Training", "ExampleFiles/EmpiricalData", 2)
    CLOUDePredictNN("ExampleFiles/Training", "ExampleFiles/EmpiricalData", 2)

    CLOUDeClassifyXGB("ExampleFiles/Training", "ExampleFiles/EmpiricalData", 4)
    CLOUDePredictXGB("ExampleFiles/Training", "ExampleFiles/EmpiricalData", 4)

    CLOUDeClassifyRF("ExampleFiles/Training", "ExampleFiles/EmpiricalData")
    CLOUDePredictRF("ExampleFiles/Training", "ExampleFiles/EmpiricalData")

    CLOUDeClassifySVM("ExampleFiles/Training", "ExampleFiles/EmpiricalData")
    CLOUDePredictSVM("ExampleFiles/Training", "ExampleFiles/EmpiricalData")

The above operations will output the files

    EmpiricalData.nn.2.classifications
    EmpiricalData.nn.2.probabilities
    EmpiricalData.nn.2.predictions

    EmpiricalData.xgb.4.classifications
    EmpiricalData.xgb.4.probabilities
    EmpiricalData.xgb.4.predictions
  
    EmpiricalData.rf.classifications
    EmpiricalData.rf.probabilities
    EmpiricalData.rf.predictions

    EmpiricalData.svm.classifications
    EmpiricalData.svm.probabilities
    EmpiricalData.svm.predictions