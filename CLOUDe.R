library(dplyr)

GenerateTrainingData <- function(m, Nk, training_prefix, logThetaMin, logThetaMax, logAlphaMin, logAlphaMax, logSigmaSqMin, logSigmaSqMax) {
  library(mvtnorm) # For multivariate Gaussian distribution 
  
  scenarios <- c("redundant", "unique")
  conditions <- seq(1, m)
  
  # m conditions with 3 expression values and 4 parameters each, and class label
  training_data <- matrix(0, nrow = 2*Nk, ncol = 1 + 7*m)
  
  for(sindex in 1:length(scenarios)) {
    scenario <- scenarios[sindex]
    
    for(i in 1:Nk) {
      tDSL <- 1
      tDS <- runif(1, 0, 1)
      
      alpha <- 10^runif(m, min = logAlphaMin, max = logAlphaMax)
      sigmaSq <- 10^runif(m, min = logSigmaSqMin, max = logSigmaSqMax)
      
      theta1 <- c()
      theta2 <- c()
      
      if(scenario == "redundant") { # REDUNDANT
        theta1 <- runif(m, min = logThetaMin, max = logThetaMax)
        theta2 <- theta1
      }
      else if(scenario == "unique") { # UNIQUE
        theta1 <- runif(m, min = logThetaMin, max = logThetaMax)
        theta2 <- runif(m, min = logThetaMin, max = logThetaMax)
      }
      
      expression_vec <- c()
      mu <- rep(0, 3)
      CovMat <- matrix(0, nrow = 3, ncol = 3)
      for(j in 1:m) {
        mu[1] = (1 - exp(-alpha[j] * tDS)) * theta1[j] + exp(-alpha[j] * tDS) * theta2[j]
        mu[2] = mu[1]
        mu[3] = theta2[j]
        CovMat[1, 1] = sigmaSq[j] / (2 * alpha[j])
        CovMat[2, 2] = CovMat[1, 1]
        CovMat[3, 3] = CovMat[1, 1]
        CovMat[1, 2] = exp(-2 * alpha[j] * tDS) * sigmaSq[j] / (2 * alpha[j])
        CovMat[2, 1] = CovMat[1, 2]
        CovMat[1, 3] = exp(-2 * alpha[j] * tDSL) * sigmaSq[j] / (2 * alpha[j])
        CovMat[3, 1] = CovMat[1, 3]
        CovMat[2, 3] = CovMat[1, 3]
        CovMat[3, 2] = CovMat[1, 3]
        expression_vec <- c(expression_vec, rmvnorm(1, mean = mu, sigma = CovMat))
      }
      
      rowIndex <- (sindex - 1)*Nk + i # get the row of the training dataset
      training_data[rowIndex, 1] = sindex
      for(j in 1:(3*m)) {
        training_data[rowIndex, 1 + j] = expression_vec[j]
      }
      for(j in 1:m) {
        training_data[rowIndex, 1 + 3*m + j] = theta1[j]
      }
      for(j in 1:m) {
        training_data[rowIndex, 1 + 4*m + j] = theta2[j]
      }
      for(j in 1:m) {
        training_data[rowIndex, 1 + 5*m + j] = log10(alpha[j])
      }
      for(j in 1:m) {
        training_data[rowIndex, 1 + 6*m + j] = log10(sigmaSq[j])
      }
    }
  }
  
  column_labels <- c("Class")
  for(j in 1:m) {
    column_labels <- c(column_labels, paste("eD", j, sep = ""), paste("eS", j, sep = ""), paste("eL", j, sep = ""))
  }
  for(j in 1:m) {
    column_labels <- c(column_labels, paste("Theta1.", j, sep = ""))
  }
  for(j in 1:m) {
    column_labels <- c(column_labels, paste("Theta2.", j, sep = ""))
  }
  for(j in 1:m) {
    column_labels <- c(column_labels, paste("LogAlpha", j, sep = ""))
  }
  for(j in 1:m) {
    column_labels <- c(column_labels, paste("LogSigmaSq", j, sep = ""))
  }
  
  colnames(training_data) <- column_labels
  
  write.table(training_data, file = paste(training_prefix, ".data", sep = ""), row.names = FALSE)
  
  GenerateFeatures(m, training_prefix)
  GenerateClassifierResponse(training_prefix)
  GeneratePredictorResponse(training_prefix)
}

GenerateFeatures <- function(m, training_prefix) {
  input_filename <- paste(training_prefix, ".data", sep = "")
  feature_filename <- paste(training_prefix, ".features", sep = "")
  
  features <- as_tibble( as.matrix(read.table(input_filename, header = TRUE)) ) %>% 
    select(starts_with("eD"), starts_with("eS"), starts_with("eL"))
  
  write.table(features, file = feature_filename, row.names = FALSE)
  
  X_means <- colMeans(features)
  X_sds <- apply(features, 2, sd)
  std_params <- data.frame("Xmeans" = X_means, "Xsds" = X_sds)
  write.table(std_params, paste(training_prefix, ".X_stdparams", sep = ""), row.names = FALSE)
  rm(std_params)
}

GenerateClassifierResponse <- function(training_prefix) {
  input_filename <- paste(training_prefix, ".data", sep = "")
  response_filename <- paste(training_prefix, ".classes", sep = "")
  
  training_data <- as_tibble( as.matrix(read.table(input_filename, header = TRUE)) ) %>% 
    select(Class) %>%
    as.matrix()
  
  response <- matrix(0, nrow = nrow(training_data), ncol = 2)
  for(i in 1:nrow(training_data)) {
    response[i, training_data[i,1]] = 1
  }
  colnames(response) <- c("redundant", "unique")
  
  write.table(response, file = response_filename, row.names = FALSE)
}

GeneratePredictorResponse <- function(training_prefix) {
  input_filename <- paste(training_prefix, ".data", sep = "")
  response_filename <- paste(training_prefix, ".responses", sep = "")
  
  preliminar_data <- as_tibble( as.matrix(read.table(input_filename, header = TRUE)) ) %>% 
    select(starts_with("theta1"), starts_with("theta2"), starts_with("LogAlpha"), starts_with("LogSigma")) %>%
    as.matrix()
  
  response <- matrix(0, nrow = nrow(preliminar_data), ncol = 18)
  column_labels <- c()
  for(i in 1:6) {
    column_labels <- c(column_labels, paste("Theta1.", i, sep = ""))
  }
  for(i in 1:6) {
    column_labels <- c(column_labels, paste("Theta2.", i, sep = ""))
  }
  for(i in 1:6) {
    column_labels <- c(column_labels, paste("SDR", i, sep = "")) # Selection-Drift ratio
  }
  colnames(response) <- column_labels
  
  for(i in 1:6){
    response[,i] <- preliminar_data[,i]
    response[,i+6] <- preliminar_data[,i+6]
    response[,i+12] <- preliminar_data[,i+12] - preliminar_data[,i+18]
  }
  
  write.table(response, file = response_filename, row.names = FALSE)
  
  Y_means <- colMeans(response)
  Y_sds <- apply(response, 2, sd)
  std_params <- data.frame("Ymeans" = Y_means, "Ysds" = Y_sds)
  write.table(std_params, paste(training_prefix, ".Y_stdparams", sep = ""), row.names = FALSE)
  rm(std_params)
}

ClassifierCVnn <- function(num_layers, batchsize, num_epochs, log_lambda_min, log_lambda_max, num_lambda, gamma_min, gamma_max, num_gamma, training_prefix) {
  if(num_layers %in% c(0, 1, 2, 3, 4, 5)) {
    library(keras)
    library(tensorflow)
    
    CV <- 5
    lambdas <- 10^seq(log_lambda_min, log_lambda_max, length = num_lambda)
    gammas <- seq(gamma_min, gamma_max, length = num_gamma)
    
    X <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
    Y <- as.matrix(read.table(paste(training_prefix, ".classes", sep = ""), header = TRUE))
    
    # standardize the input for training
    X_means <- colMeans(X)
    X_sds <- apply(X, 2, sd)
    for(j in 1:ncol(X)) {
      X[,j] = (X[,j] - X_means[j]) / X_sds[j]
    }
    
    # CV-fold cross validation
    val_loss <- array(0, dim = c(length(gammas), length(lambdas)))
    
    # Randomly choose balanced training/validation sets per fold
    foldid_redundant <- sample(rep(seq(CV), length = nrow(X)/2))
    foldid_unique <- sample(rep(seq(CV), length = nrow(X)/2))
    foldid <- c(foldid_redundant, foldid_unique)
    rm(foldid_redundant)
    rm(foldid_unique)
    
    # Perform K-fold CV, where K = CV
    for(curr_fold in 1:CV) {
      Xval <- X[foldid == curr_fold, ]
      Xtrain <- X[foldid != curr_fold, ]
      Yval <- Y[foldid == curr_fold, ]
      Ytrain <- Y[foldid != curr_fold, ]
      
      # standardize the input for train and val based on train
      temp_means <- colMeans(Xtrain)
      temp_sds <- apply(Xtrain, 2, sd)
      for(j in 1:ncol(Xtrain)) {
        Xtrain[,j] = (Xtrain[,j] - temp_means[j]) / temp_sds[j]
        Xval[,j] = (Xval[,j] - temp_means[j]) / temp_sds[j]
      }
      
      for(i in 1:length(gammas)) {
        for(j in 1:length(lambdas)) {
          model <- keras_model_sequential()
          if(num_layers == 0) {
            model %>%
              layer_dense(units = 2,
                          activation = 'softmax',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain)))
          }
          else if(num_layers == 1) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 2,
                          activation = 'softmax',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 2) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 2,
                          activation = 'softmax',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 3) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 64, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 2,
                          activation = 'softmax',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 4) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 64, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 32, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 2,
                          activation = 'softmax',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 5) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 64, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 32, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 16, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 2,
                          activation = 'softmax',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          
          model %>% 
            compile(loss = 'categorical_crossentropy',
                    optimizer = optimizer_adam(),
                    metrics = c('categorical_crossentropy'))
          
          history <- model %>% 
            fit(Xtrain, Ytrain,
                epochs = num_epochs,
                batch_size = batchsize,
                validation_data = list(Xval, Yval),
                verbose = 0) # verbose = 0  ensures it is silent
          
          val_loss[i,j] <- val_loss[i,j] + as.data.frame(history) %>%
            filter(data == "validation", metric == "categorical_crossentropy") %>%
            select(value) %>%
            min()
        }
      }
      
      rm(Xval)
      rm(Xtrain)
      rm(Yval)
      rm(Ytrain)
    }
    
    val_loss <- val_loss / CV
    
    gamma_opt = gammas[ which(val_loss == min(val_loss), arr.ind = TRUE)[1] ]
    lambda_opt = lambdas[ which(val_loss == min(val_loss), arr.ind = TRUE)[2] ]
    cv_results <- matrix(0, 1, 3)
    cv_results[,1] = min(val_loss)
    cv_results[,2] = gamma_opt
    cv_results[,3] = lambda_opt
    colnames(cv_results) <- c("Loss", "Gamma", "Lambda")
    
    write.table(cv_results, file = paste(training_prefix, ".nn.", num_layers, ".classifier_cv", sep = ""), row.names = FALSE)
    
    ClassifierFitNN(num_layers, batchsize, num_epochs, training_prefix)
  }
}

ClassifierFitNN <- function(num_layers, batchsize, num_epochs, training_prefix) {
  library(keras)
  library(tensorflow)
  
  cv_results <- read.table(paste(training_prefix, ".nn.", num_layers, ".classifier_cv", sep = ""), header = TRUE)
  gamma_opt <- cv_results$Gamma
  lambda_opt <- cv_results$Lambda
  
  X <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
  Y <- as.matrix(read.table(paste(training_prefix, ".classes", sep = ""), header = TRUE))
  
  # standardize the input for training
  X_means <- colMeans(X)
  X_sds <- apply(X, 2, sd)
  for(j in 1:ncol(X)) {
    X[,j] = (X[,j] - X_means[j]) / X_sds[j]
  }
  
  model <- keras_model_sequential()
  if(num_layers == 0) {
    model %>%
      layer_dense(units = 2,
                  activation = 'softmax',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X)))
  }
  else if(num_layers == 1) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 2,
                  activation = 'softmax',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 2) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 2,
                  activation = 'softmax',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 3) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 64, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 2,
                  activation = 'softmax',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 4) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 64, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 32, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 2,
                  activation = 'softmax',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 5) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 64, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 32, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 16, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 2,
                  activation = 'softmax',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  
  model %>% 
    compile(loss = 'categorical_crossentropy',
            optimizer = optimizer_adam(),
            metrics = c('categorical_crossentropy', 'accuracy'))
  
  history <- model %>% 
    fit(X, Y,
        epochs = num_epochs,
        batch_size = batchsize,
        verbose = 0) # verbose = 0  ensures it is silent
  
  model %>% save_model_hdf5(paste(training_prefix, ".nn.", num_layers, ".classifier.hdf5", sep = ""))
}

PredictorCVnn <- function(num_layers, batchsize, num_epochs, log_lambda_min, log_lambda_max, num_lambda, gamma_min, gamma_max, num_gamma, training_prefix) {
  if(num_layers %in% c(0, 1, 2, 3, 4, 5)) {
    library(keras)
    library(tensorflow)
    
    CV <- 5
    lambdas <- 10^seq(log_lambda_min, log_lambda_max, length = num_lambda)
    gammas <- seq(gamma_min, gamma_max, length = num_gamma)
    
    X <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
    Y <- as.matrix(read.table(paste(training_prefix, ".responses", sep = ""), header = TRUE))
    
    # standardize the input for training
    X_means <- colMeans(X)
    X_sds <- apply(X, 2, sd)
    Y_means <- colMeans(Y)
    Y_sds <- apply(Y, 2, sd)
    for(j in 1:ncol(X)) {
      X[,j] = (X[,j] - X_means[j]) / X_sds[j]
    }
    for(j in 1:ncol(Y)) {
      Y[,j] = (Y[,j] - Y_means[j]) / Y_sds[j]
    }
    
    # CV-fold cross validation
    val_loss <- array(0, dim = c(length(gammas), length(lambdas)))
    
    # Randomly choose balanced training/validation sets per fold
    foldid_redundant <- sample(rep(seq(CV), length = nrow(X)/2))
    foldid_unique <- sample(rep(seq(CV), length = nrow(X)/2))
    foldid <- c(foldid_redundant, foldid_unique)
    rm(foldid_redundant)
    rm(foldid_unique)
    
    # Perform K-fold CV, where K = CV
    for(curr_fold in 1:CV) {
      Xval <- X[foldid == curr_fold, ]
      Xtrain <- X[foldid != curr_fold, ]
      Yval <- Y[foldid == curr_fold, ]
      Ytrain <- Y[foldid != curr_fold, ]
      
      # standardize the input for train and val based on train
      temp_Xmeans <- colMeans(Xtrain)
      temp_Xsds <- apply(Xtrain, 2, sd)
      temp_Ymeans <- colMeans(Ytrain)
      temp_Ysds <- apply(Ytrain, 2, sd)
      for(j in 1:ncol(Xtrain)) {
        Xtrain[,j] = (Xtrain[,j] - temp_Xmeans[j]) / temp_Xsds[j]
        Xval[,j] = (Xval[,j] - temp_Xmeans[j]) / temp_Xsds[j]
      }
      for(j in 1:ncol(Ytrain)) {
        Ytrain[,j] = (Ytrain[,j] - temp_Ymeans[j]) / temp_Ysds[j]
        Yval[,j] = (Yval[,j] - temp_Ymeans[j]) / temp_Ysds[j]
      }
      
      for(i in 1:length(gammas)) {
        for(j in 1:length(lambdas)) {
          model <- keras_model_sequential()
          if(num_layers == 0) {
            model %>%
              layer_dense(units = ncol(Ytrain),
                          activation = 'linear',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain)))
          }
          else if(num_layers == 1) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = ncol(Ytrain),
                          activation = 'linear',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 2) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = ncol(Ytrain),
                          activation = 'linear',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 3) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 64, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = ncol(Ytrain),
                          activation = 'linear',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 4) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 64, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 32, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = ncol(Ytrain),
                          activation = 'linear',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          else if(num_layers == 5) {
            model %>%
              layer_dense(units = 256, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]),
                          input_shape = c(ncol(Xtrain))) %>%
              layer_dense(units = 128, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 64, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 32, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = 16, 
                          activation = 'relu',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j])) %>%
              layer_dense(units = ncol(Ytrain),
                          activation = 'linear',
                          kernel_regularizer = regularizer_l1_l2(l1 = gammas[i] * lambdas[j], l2 = (1 - gammas[i]) * lambdas[j]))
          }
          
          model %>% 
            compile(loss = 'mean_squared_error',
                    optimizer = optimizer_adam(),
                    metrics = c('mean_squared_error'))
          
          history <- model %>% 
            fit(Xtrain, Ytrain,
                epochs = num_epochs,
                batch_size = batchsize,
                validation_data = list(Xval, Yval),
                verbose = 0) # verbose = 0  ensures it is silent
          
          val_loss[i,j] <- val_loss[i,j] + as.data.frame(history) %>%
            filter(data == "validation", metric == "mean_squared_error") %>%
            select(value) %>%
            min()
        }
      }
      
      rm(Xval)
      rm(Xtrain)
      rm(Yval)
      rm(Ytrain)
    }
    
    val_loss <- val_loss / CV
    
    gamma_opt = gammas[ which(val_loss == min(val_loss), arr.ind = TRUE)[1] ]
    lambda_opt = lambdas[ which(val_loss == min(val_loss), arr.ind = TRUE)[2] ]
    cv_results <- matrix(0, 1, 3)
    cv_results[,1] = min(val_loss)
    cv_results[,2] = gamma_opt
    cv_results[,3] = lambda_opt
    colnames(cv_results) <- c("Loss", "Gamma", "Lambda")
    
    write.table(cv_results, file = paste(training_prefix, ".nn.", num_layers, ".predictor_cv", sep = ""), row.names = FALSE)
    
    PredictorFitNN(num_layers, batchsize, num_epochs, training_prefix)
  }
}

PredictorFitNN <- function(num_layers, batchsize, num_epochs, training_prefix) {
  library(keras)
  library(tensorflow)
  
  cv_results <- read.table(paste(training_prefix, ".nn.", num_layers, ".predictor_cv", sep = ""), header = TRUE)
  gamma_opt <- cv_results$Gamma
  lambda_opt <- cv_results$Lambda
  
  X <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
  Y <- as.matrix(read.table(paste(training_prefix, ".responses", sep = ""), header = TRUE))
  
  # standardize the input and output for training
  X_means <- colMeans(X)
  X_sds <- apply(X, 2, sd)
  Y_means <- colMeans(Y)
  Y_sds <- apply(Y, 2, sd)
  for(j in 1:ncol(X)) {
    X[,j] = (X[,j] - X_means[j]) / X_sds[j]
  }
  for(j in 1:ncol(Y)) {
    Y[,j] = (Y[,j] - Y_means[j]) / Y_sds[j]
  }
  
  model <- keras_model_sequential()
  if(num_layers == 0) {
    model %>%
      layer_dense(units = ncol(Y),
                  activation = 'linear',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X)))
  }
  else if(num_layers == 1) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = ncol(Y),
                  activation = 'linear',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 2) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = ncol(Y),
                  activation = 'linear',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 3) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 64, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = ncol(Y),
                  activation = 'linear',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 4) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 64, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 32, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = ncol(Y),
                  activation = 'linear',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  else if(num_layers == 5) {
    model %>%
      layer_dense(units = 256, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt),
                  input_shape = c(ncol(X))) %>%
      layer_dense(units = 128, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 64, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 32, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = 16, 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt)) %>%
      layer_dense(units = ncol(Y),
                  activation = 'linear',
                  kernel_regularizer = regularizer_l1_l2(l1 = gamma_opt * lambda_opt, l2 = (1 - gamma_opt) * lambda_opt))
  }
  
  model %>% 
    compile(loss = 'mean_squared_error',
            optimizer = optimizer_adam(),
            metrics = 'mean_squared_error')
  
  history <- model %>% 
    fit(X, Y,
        epochs = num_epochs,
        batch_size = batchsize,
        verbose = 0) # verbose = 0 ensures it is silent
  
  model %>% save_model_hdf5(paste(training_prefix, ".nn.", num_layers, ".predictor.hdf5", sep = ""))
}

CLOUDeClassifyNN <- function(training_prefix, testing_prefix, num_layers) {
  if(num_layers %in% c(0, 1, 2, 3, 4, 5)) {
    library(keras)
    library(tensorflow)
    
    X <- as.matrix(read.table(paste(testing_prefix, ".features", sep = ""), header = TRUE))
    
    # standardize the input for testing
    std_params <- read.table(paste(training_prefix, ".X_stdparams", sep = ""), header = TRUE)
    X_means <- c(std_params$Xmeans)
    X_sds <- c(std_params$Xsds)
    for(j in 1:ncol(X)) {
      X[,j] = (X[,j] - X_means[j]) / X_sds[j]
    }
    rm(std_params)
    
    model <- load_model_hdf5(paste(training_prefix, ".nn.", num_layers, ".classifier.hdf5", sep = ""))
    
    Yest_num <- model %>% predict(X) %>% k_argmax()
    
    Yest <- data.frame("Class" = ifelse(Yest_num == 0, "redundant", "unique"))
    
    write.table(Yest, paste(testing_prefix, ".nn.", num_layers, ".classifications", sep = ""), row.names = FALSE)
    
    probs_est <- model %>% predict(X)
    colnames(probs_est) <- c("redundant", "unique")
    
    write.table(probs_est, paste(testing_prefix, ".nn.", num_layers, ".probabilities", sep = ""), row.names = FALSE)
  }
}

CLOUDeClassifySVM <- function(training_prefix, testing_prefix) {
  library(liquidSVM) # For SVM
  
  Xtrain <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
  Ytrain <- read.table(paste(training_prefix, ".classes", sep = ""), header = TRUE) %>%
    transmute(class = as.factor(ifelse(redundant == 1, "redundant", "unique"))) 
  X <- as.matrix(read.table(paste(testing_prefix, ".features", sep = ""), header = TRUE))
  
  # standardize the input for training and testing
  std_params <- read.table(paste(training_prefix, ".X_stdparams", sep = ""), header = TRUE)
  X_means <- c(std_params$Xmeans)
  X_sds <- c(std_params$Xsds)
  for(j in 1:ncol(X)) {
    Xtrain[,j] = (Xtrain[,j] - X_means[j]) / X_sds[j]
    X[,j] = (X[,j] - X_means[j]) / X_sds[j]
  }
  rm(std_params)
  
  svm_trained <- mcSVM(x = Xtrain, y = Ytrain$class, mc_type = "OvA_hinge", folds = 5, scale = FALSE, do.select = TRUE)
  preds <- c(predict(svm_trained, X))
  
  Yest <- data.frame("Class" = ifelse(preds <= 0.0, "redundant", "unique"))
  
  write.table(Yest, paste(testing_prefix, ".svm.classifications", sep = ""), row.names = FALSE)

  pred_probs <- (preds + 1) / 2
  probs_est <- data.frame("redundant" = 1 - pred_probs, "unique" = pred_probs)
  
  write.table(probs_est, paste(testing_prefix, ".svm.probabilities", sep = ""), row.names = FALSE)
}

CLOUDeClassifyRF <- function(training_prefix, testing_prefix) {
  library(ranger) # For RF
  
  Xtrain <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
  Ytrain <- read.table(paste(training_prefix, ".classes", sep = ""), header = TRUE) %>%
    transmute(class = as.factor(ifelse(redundant == 1, "redundant", "unique"))) 
  X <- as.matrix(read.table(paste(testing_prefix, ".features", sep = ""), header = TRUE))
  
  # standardize the input for training and testing
  std_params <- read.table(paste(training_prefix, ".X_stdparams", sep = ""), header = TRUE)
  X_means <- c(std_params$Xmeans)
  X_sds <- c(std_params$Xsds)
  for(j in 1:ncol(X)) {
    Xtrain[,j] = (Xtrain[,j] - X_means[j]) / X_sds[j]
    X[,j] = (X[,j] - X_means[j]) / X_sds[j]
  }
  rm(std_params)
  
  tempData <- as.data.frame( cbind(Xtrain, Ytrain) )
  rf_trained <- ranger(class ~., data = tempData, probability = TRUE)
  preds <- predict(rf_trained, X)

  probs_est <- as.data.frame(preds$predictions)
  Yest <- data.frame("Class" = ifelse(probs_est$redundant > 0.5, "redundant", "unique"))
  
  write.table(Yest, paste(testing_prefix, ".rf.classifications", sep = ""), row.names = FALSE)
  write.table(probs_est, paste(testing_prefix, ".rf.probabilities", sep = ""), row.names = FALSE)
}

CLOUDePredictNN <- function(training_prefix, testing_prefix, num_layers) {
  if(num_layers %in% c(0, 1, 2, 3, 4, 5)) {
    library(keras)
    library(tensorflow)
    
    X <- as.matrix(read.table(paste(testing_prefix, ".features", sep = ""), header = TRUE))
    
    # standardize the input for testing
    std_params <- read.table(paste(training_prefix, ".X_stdparams", sep = ""), header = TRUE)
    X_means <- c(std_params$Xmeans)
    X_sds <- c(std_params$Xsds)
    for(j in 1:ncol(X)) {
      X[,j] = (X[,j] - X_means[j]) / X_sds[j]
    }
    rm(std_params)
    
    model <- load_model_hdf5(paste(training_prefix, ".nn.", num_layers, ".predictor.hdf5", sep = ""))
    
    std_params <- read.table(paste(training_prefix, ".Y_stdparams", sep = ""), header = TRUE)
    Y_means <- c(std_params$Ymeans)
    Y_sds <- c(std_params$Ysds)
    rm(std_params)
    
    Yest_std <- model %>% predict(X)
    Yest <- Yest_std
    
    # un-standardize the responses
    for(j in 1:ncol(Yest)) {
      Yest[, j] = Yest_std[,j] * Y_sds[j] + Y_means[j]
    }
    
    Y <- as.matrix(read.table(paste(training_prefix, ".responses", sep = ""), header = TRUE))
    colnames(Yest) <- colnames(Y)
    rm(Y)
    
    write.table(Yest, paste(testing_prefix, ".nn.", num_layers, ".predictions", sep = ""), row.names = FALSE)
  }
}

CLOUDePredictSVM <- function(training_prefix, testing_prefix) {
  library(liquidSVM) # For SVM

  Xtrain <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
  Ytrain <- as.matrix(read.table(paste(training_prefix, ".responses", sep = ""), header = TRUE))
  X <- as.matrix(read.table(paste(testing_prefix, ".features", sep = ""), header = TRUE))
  
  # standardize the input for training and testing
  std_params <- read.table(paste(training_prefix, ".X_stdparams", sep = ""), header = TRUE)
  X_means <- c(std_params$Xmeans)
  X_sds <- c(std_params$Xsds)
  for(j in 1:ncol(X)) {
    Xtrain[,j] = (Xtrain[,j] - X_means[j]) / X_sds[j]
    X[,j] = (X[,j] - X_means[j]) / X_sds[j]
  }
  rm(std_params)
  
  std_params <- read.table(paste(training_prefix, ".Y_stdparams", sep = ""), header = TRUE)
  Y_means <- c(std_params$Ymeans)
  Y_sds <- c(std_params$Ysds)
  for(j in 1:ncol(Ytrain)) {
    Ytrain[,j] = (Ytrain[,j] - Y_means[j]) / Y_sds[j]
  }
  rm(std_params)
  
  Yest <- matrix(0, nrow = nrow(X), ncol = ncol(Ytrain))
  for(i in 1:ncol(Ytrain)) {
    svm_trained <- lsSVM(x = Xtrain, y = Ytrain[,i], folds = 5, scale = FALSE, do.select = TRUE)
    Yest[,i] <- c(predict(svm_trained, X))
  }
  
  # un-standardize the responses
  for(j in 1:ncol(Yest)) {
    Yest[, j] = Yest[,j] * Y_sds[j] + Y_means[j]
  }
  
  colnames(Yest) <- colnames(Ytrain)
  rm(Ytrain)
  
  write.table(Yest, paste(testing_prefix, ".svm.predictions", sep = ""), row.names = FALSE)
}

CLOUDePredictRF <- function(training_prefix, testing_prefix) {
  library(ranger) # For RF
  
  Xtrain <- as.matrix(read.table(paste(training_prefix, ".features", sep = ""), header = TRUE))
  Ytrain <- as.matrix(read.table(paste(training_prefix, ".responses", sep = ""), header = TRUE))
  X <- as.matrix(read.table(paste(testing_prefix, ".features", sep = ""), header = TRUE))
  
  # standardize the input for training and testing
  std_params <- read.table(paste(training_prefix, ".X_stdparams", sep = ""), header = TRUE)
  X_means <- c(std_params$Xmeans)
  X_sds <- c(std_params$Xsds)
  for(j in 1:ncol(X)) {
    Xtrain[,j] = (Xtrain[,j] - X_means[j]) / X_sds[j]
    X[,j] = (X[,j] - X_means[j]) / X_sds[j]
  }
  rm(std_params)
  
  std_params <- read.table(paste(training_prefix, ".Y_stdparams", sep = ""), header = TRUE)
  Y_means <- c(std_params$Ymeans)
  Y_sds <- c(std_params$Ysds)
  for(j in 1:ncol(Ytrain)) {
    Ytrain[,j] = (Ytrain[,j] - Y_means[j]) / Y_sds[j]
  }
  rm(std_params)
  
  Yest <- matrix(0, nrow = nrow(X), ncol = ncol(Ytrain))
  for(i in 1:ncol(Ytrain)) {
    tempData <- as.data.frame( cbind(Xtrain, Ytrain[,i]) )
    colnames(tempData) <- c(colnames(Xtrain), "regResp")
    rf_trained <- ranger(regResp ~., data = tempData)
    Yest[,i] <- c(predict(rf_trained, X)$predictions)
  }
  
  # un-standardize the responses
  for(j in 1:ncol(Yest)) {
    Yest[, j] = Yest[,j] * Y_sds[j] + Y_means[j]
  }
  
  colnames(Yest) <- colnames(Ytrain)
  rm(Ytrain)
  
  write.table(Yest, paste(testing_prefix, ".rf.predictions", sep = ""), row.names = FALSE)
}