library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)


# Load libraries
library(xgboost)
library(caret)

# Define sample sizes
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

# Initialize result dataframe
results <- data.frame(Method = character(), 
                      DatasetSize = integer(), 
                      Accuracy = numeric(), 
                      TimeTakenSecs = numeric())

for (sz in sizes) {
  # Sample the data
  sampled_data <- dfdata[sample(1:nrow(dfdata), size = sz, replace = TRUE), ]
  
  # Split into training and testing sets
  trainIndex <- createDataPartition(sampled_data$outcome, p = 0.8, list = FALSE)
  trainData <- sampled_data[trainIndex, ]
  testData <- sampled_data[-trainIndex, ]
  
  # Prepare xgboost matrices
  dtrain <- xgb.DMatrix(data = as.matrix(trainData[, -ncol(trainData)]), label = trainData$outcome)
  dtest <- xgb.DMatrix(data = as.matrix(testData[, -ncol(testData)]), label = testData$outcome)
  
  # Record start time
  start_time <- Sys.time()
  
  # Train xgboost model
  model <- xgboost(data = dtrain,
                   objective = "binary:logistic",
                   nrounds = 50,
                   verbose = 0)
  
  # Record end time
  end_time <- Sys.time()
  time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Make predictions
  preds <- predict(model, dtest)
  pred_labels <- ifelse(preds > 0.5, 1, 0)
  
  # Calculate accuracy
  acc <- mean(pred_labels == testData$outcome)
  
  # Save results
  results <- rbind(results, 
                   data.frame(Method = "XGBoost (simple CV)",
                              DatasetSize = sz,
                              Accuracy = acc,
                              TimeTakenSecs = time_taken))
  
  print(paste("Finished size:", sz))
}

print(results)


# Load libraries
library(caret)
library(xgboost)
# Define sample sizes
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

# Initialize result dataframe
results <- data.frame(Method = character(),
                      DatasetSize = integer(),
                      Accuracy = numeric(),
                      TimeTakenSecs = numeric())

# Control for caret 5-fold CV
fitControl <- trainControl(method = "cv", number = 5)

for (sz in sizes) {
  # Sample the data
  sampled_data <- dfdata[sample(1:nrow(dfdata), size = sz, replace = TRUE), ]
  
  # Split predictors and outcome
  x <- sampled_data[, -ncol(sampled_data)]
  y <- as.factor(sampled_data$outcome)  # caret expects factors for classification
  
  # Record start time
  start_time <- Sys.time()
  
  # Train using caret with xgboost method
  model <- train(x = x,
                 y = y,
                 method = "xgbTree",
                 trControl = fitControl,
                 verbose = FALSE)
  
  # Record end time
  end_time <- Sys.time()
  time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Get best cross-validated accuracy
  acc <- max(model$results$Accuracy)
  
  # Save results
  results <- rbind(results,
                   data.frame(Method = "XGBoost (caret 5-fold CV)",
                              DatasetSize = sz,
                              Accuracy = acc,
                              TimeTakenSecs = time_taken))
  
  print(paste("Finished size:", sz))
}

print(results)




