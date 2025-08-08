get_QDA <- function(x1, x2, y, new1, new2) {

#Step 1:Input Validation Checks
  # In this step,I have checked that the inputs are correct before running the QDA Function.
  # I make sure that x1,x2.new1,new2 are numbers and the "y" is either character or factor type.
  # I also check that all input vectors are the same length. 
  # Type checks
  if (!is.numeric(x1) || !is.numeric(x2)) stop("x1 and x2 must be numeric.")
  if (!is.character(y) && !is.factor(y)) stop("y must be a character or factor vector.")
  if (!is.numeric(new1) || !is.numeric(new2)) stop("new1 and new2 must be numeric.")
  
  # Length checks
  if (length(x1) != length(x2) || length(x1) != length(y)) stop("x1, x2, and y must have the same length.")
  if (length(new1) != length(new2)) stop("new1 and new2 must have the same length.")
  
# step 2:Combine data
  # In this step I have combined the input vectors into data frames.
  # The x1,x2 and y values are combined into one data set called data,which I use to calculate class statistics.
  # The new1 and new2 values are combined into another data set called new_data,This contains the new observation that is needed for classification of QDA.
  
  data <- data.frame(x1 = x1, x2 = x2, class = y)
  new_data <- data.frame(x1 = new1, x2 = new2)

#Step 3: Edge Case Validations
  # In this step I have checked for the special cases that could break the QDA model.I make sure that
  # there are at least two different classes in y, each class has at least two data points,
  #each predictor has some variation, and the covariance matrix can be calculated.

  classes <- unique(y)
  
  if (length(classes) < 2) stop("y must contain at least two distinct classes.")
  
  for (cls in classes) {
    subset_k <- data[data$class == cls, c("x1", "x2")]
    
    if (nrow(subset_k) < 2) stop(paste("Class", cls, "must have at least two observations."))
    if (any(apply(subset_k, 2, var) == 0)) stop(paste("Class", cls, "must have non-zero variance in both predictors."))
    if (det(cov(subset_k)) == 0) stop(paste("Covariance matrix for class", cls, "is not invertible."))
  }
  
# Step 4; QDA Calculation
  # In this step, I have calculated how likely each new observation belongs to each class.
  # For every class I find the mean, covariance and prior probability.Then, for each new point,
  # I used the QDA formula to calculate the score that measures how well the point fits that class. These scores are stored
  # in a matrix so I can compare them later.

  scores <- matrix(NA, nrow = nrow(new_data), ncol = length(classes))
  colnames(scores) <- classes
  
  for (k in seq_along(classes)) {
    class_k <- classes[k]
    subset_k <- data[data$class == class_k, c("x1", "x2")]
    
    mu_k <- colMeans(subset_k)
    sigma_k <- cov(subset_k)
    pi_k <- nrow(subset_k) / nrow(data)
    
    inv_sigma_k <- solve(sigma_k)
    det_sigma_k <- det(sigma_k)
    
    for (i in 1:nrow(new_data)) {
      x_vec <- as.numeric(new_data[i, ])
      diff <- x_vec - mu_k
      
      # Discriminant function
      delta <- -0.5 * log(det_sigma_k) -
        0.5 * t(diff) %*% inv_sigma_k %*% diff +
        log(pi_k)
      
      scores[i, k] <- delta
    }
  }

# Step 5:Return Predicted Classes
  # In this step,we look at the QDA scores for each new observation and pick the class
  # with the highest scores.
  preds <- apply(scores, 1, function(row) names(which.max(row)))
  return(preds)
}

# testing the get_QDA function;
# In this step I have 9 data points split evenly into 3 groups A,B,C each point
# Each point has two values(Xx1 and x2) that describe it.Then I used get_QDA() function
# to predict the class for two new points(4,5) and (2,1).
# The function compares each new point to the patterns in each group and picks the one it fits best.

x1 <- c(1,6,7,0,2,5,8,3,4)
x2 <- 0:8
y <- rep(LETTERS[1:3], each = 3)
new1 <- c(4,2)
new2 <- c(5,1)

get_QDA(x1, x2, y, new1, new2)

