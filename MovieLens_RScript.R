
##########################################################
# Capstone - MovieLens: R-Sricpt (Martin Knell)
##########################################################


##########################################################
# Load Libraries
##########################################################

library(tidyverse)
library(caret)
library(data.table)
library(rmarkdown)
library(lubridate)


##########################################################
# Create Data Set (Code Provided)
##########################################################

# User provided code. Code creates subset of 10M rows from the MovieLens data set. 
# Code creates a training set (edx) and a validation set (validation) containing 10% of the data. 

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Data Set Exploration
##########################################################

# Head and Variables, columns, rows

head(edx)
nrow(edx)  # A. 9000055 (Approx. 9M rows in training data set)
ncol(edx)  # A. 6 (6 variables/ columns)


# Number movies and users

movies <- unique(edx$movieId)
length(movies)  # A. 10677  (Over ten thousand movies in data set)

users <- unique(edx$userId)
length(users)   # A. 69878  (Nearly 70,000 users in data set)


# Summary information

summary(edx)  # A. No missing values; Ratings from 0.5 to 5


# Ratings by genre

sum(str_detect(edx$genres, "Drama"))      # A. 3910127
sum(str_detect(edx$genres, "Comedy"))     # A. 3540930
sum(str_detect(edx$genres, "Thriller"))   # A. 2325899
sum(str_detect(edx$genres, "Romance"))    # A. 1712100


# Rating Frequency

rating_count <- edx %>% group_by(rating) %>% summarize(n = n())
rating_count[order(rating_count$n),]  # Most common rating is 4, followed by 3; Least common 0.5

half_star <- rating_count %>% filter(rating %in% c(0.5, 1.5, 2.5, 3.5, 4.5))
full_star <- rating_count %>% filter(rating %in% c(1,2,3,4,5))

sum(half_star$n)
sum(full_star$n)    # Full star ratings are more common than half star ratings

  # histogram on rating frequency

rating_plot <- edx %>% ggplot(aes(rating)) + geom_histogram(binwidth = 0.25, color = "blue", fill = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  theme_classic(base_size = 15) + ggtitle("Rating Score Frequency")
rating_plot

# Number ratings by movie

movie_count <- edx %>% group_by(movieId) %>% summarize(n = n())
movie_count[order(-movie_count$n), ]


# Three movies with most ratings (all above 30,000 ratings)

edx$title[edx$movieId == 296]   # A. Pulp Fiction
edx$title[edx$movieId == 356]   # A. Forest Gump
edx$title[edx$movieId == 593]   # A. The Silence of the Lambs


# Percent of movies with just one ratings, over 100, over 1000, over 10000

mean(movie_count$n == 1)       # A. 11.8%
mean(movie_count$n >= 100)     # A. 53.5%
mean(movie_count$n >= 1000)    # A. 17.8%
mean(movie_count$n >= 10000)   # A. 1.3%


# Movie effect on rating

movie_effect <- edx %>% group_by(movieId) %>% summarize(avg_score = mean(rating)) %>% ggplot(aes(avg_score)) +
  geom_histogram(binwidth = 0.25, color = "black", fill = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +scale_y_continuous(breaks = c(seq(0, 1500, 500))) +
  theme_classic(base_size = 15) + ggtitle("Movie effect of Ratings") + xlab("average Score") + ylab("# movies")
movie_effect

  # average movie rating for reference

mean(edx$rating)   # A. 3.5


# User effect on rating (users with at least 100 ratings)

user_effect <- edx %>% group_by(userId) %>% summarize(avg_score = mean(rating), n = n()) %>% filter(n >= 100) %>%
  ggplot(aes(avg_score)) +
  geom_histogram(binwidth = 0.25, color = "black", fill = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +scale_y_continuous(breaks = c(seq(0, 10000, 1000))) +
  theme_classic(base_size = 15) + ggtitle("User effect of Ratings") + xlab("average Score") + ylab("# movies")
user_effect


## Time effect on rating / Plot average weekly rating against date

time_effect <- edx %>% mutate(date = as_datetime(timestamp)) %>% mutate(week = round_date(date, unit = "week")) %>%
  group_by(week) %>% summarize(avg = mean(rating)) %>% ggplot(aes(week, avg)) + geom_point() + geom_smooth() +
  ylim(3.25,3.75) + ylab("Avg rating") + ggtitle("Time effect of ratings") + theme_classic(base_size = 15)
time_effect

#########################################
# Model Building - Movie and User Effect
#########################################

# Evaluation criteria RMSE (Root Mean Squared Error)

RMSE <- function(true_ratings, predicted_ratings){sqrt(mean((true_ratings - predicted_ratings)^2))}


# Effect of predicting average rating

avg <- mean(edx$rating)
avg

rmse_1 <- RMSE(validation$rating, avg)
rmse_1  # A. 1.061202

rmses <- data_frame(method = "Predict Average", RMSE = rmse_1)
rmses %>% knitr::kable()


# Improve by adding movie effect

movie_effect <- edx %>% group_by(movieId) %>% summarize(bi = mean(rating - avg))
predicted_2 <- avg + validation %>% left_join(movie_effect, by = 'movieId') %>% .$bi

rmse_2 <- RMSE(predicted_2, validation$rating)
rmse_2  # A. 0.94

rmses <- bind_rows(rmses, data_frame(method = "Add Movie Effect", RMSE = rmse_2))
rmses %>% knitr::kable()


# Improve further by adding user effect

user_effect <- edx %>% left_join(movie_effect, by = 'movieId') %>% group_by(userId) %>%
  summarize(bu = mean(rating - avg - bi))

predicted_3 <- validation %>% left_join(movie_effect, by = 'movieId') %>% left_join(user_effect, by = 'userId') %>%
  mutate(pred_3 = avg + bi + bu) %>% .$pred_3

rmse_3 <- RMSE(predicted_3, validation$rating)
rmse_3  # A. 0.865

rmses <- bind_rows(rmses, data_frame(method = "Add User Effect", RMSE = rmse_3))
rmses %>% knitr::kable()
                   

#########################################
# Regularization
#########################################

# Introduce penalty term to penalize large estimates that come from small sample sizes
# use lambda as a tuning parameter as in course
# For each lambda find bi, bu and rating prediction

lambdas <- seq(0,10,0.1)

rmse_lambda <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  bi <- edx %>% group_by(movieId) %>% summarize(bi = sum(rating - mu)/(n()+l))
  bu <- edx %>% left_join(bi, by="movieId") %>% group_by(userId) %>% summarize(bu = sum(rating - bi - mu)/(n()+l))
  
  predicted_ratings <- validation %>% left_join(bi, by = "movieId") %>% left_join(bu, by = "userId") %>%
    mutate(pred = mu + bi + bu) %>% .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Find optimal lambda

qplot(lambdas, rmse_lambda)

lambda <- lambdas[which.min(rmse_lambda)]
lambda   # A. 5.2

rmse_4 <- min(rmse_lambda)
rmse_4  # A. 0.8648170

rmses <- bind_rows(rmses, data_frame(method = "Regularization", RMSE = rmse_4))
rmses %>% knitr::kable()


#########################################
# Alternative 1: Gradient Boosting
#########################################


# Grid Search (w. 3-fold cross validation)

fitControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1)
gbmGrid <-  expand.grid(interaction.depth = c(3,5,7), n.trees = c(100,300,500), shrinkage = c(0.1, 0.01), n.minobsinnode = 10)

rating_pred <- train(rating ~ movieId + userId, data = edx, method = "gbm", trControl = fitControl, tuneGrid = gbmGrid)

forecast <- predict(rating_pred, validation)

summary(rating_pred)
plot(rating_pred)  # Optimal: Shrinkage 0.1, interaction.depth 7, 500 trees

rmse_5 <- RMSE(forecast, validation$rating)
rmse_5  # A. 0.9991536

rmses <- bind_rows(rmses, data_frame(method = "GBM Initial Grid", RMSE = rmse_5))
rmses %>% knitr::kable()


# New Run with improved parameters (interaction 9, trees 3000, shrinkage 0.1)

fitControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1)
gbmGrid <-  expand.grid(interaction.depth = 9, n.trees = 3000, shrinkage = 0.1, n.minobsinnode = 10)

rating_pred <- train(rating ~ movieId + userId, data = edx, method = "gbm", trControl = fitControl, tuneGrid = gbmGrid)

forecast <- predict(rating_pred, validation)

summary(rating_pred)
plot(rating_pred)  

rmse_6 <- RMSE(forecast, validation$rating)
rmse_6

rmses <- bind_rows(rmses, data_frame(method = "GBM 3000 Trees", RMSE = rmse_6))
rmses %>% knitr::kable()


#########################################
# Alternative 2: Knn
#########################################

## Note: Couldn't complete due to run time on laptop, k = 25

rating_pred <- train(rating ~ movieId + userId, data = edx, method = "knn", 
                     tuneGrid = expand.grid(k = 25))

forecast <- predict(rating_pred, validation)

rmse_7 <- RMSE(forecast, validation$rating)
rmse_7  

rating_pred$bestTune

rmses_ml <- data_frame(method = "Knn model", RMSE = rmse_7)
rmses_ml %>% knitr::kable()

