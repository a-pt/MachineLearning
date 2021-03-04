df <- read.csv("D:/Online courses/ML(Rstudio)/resources/Complete ML in R/2. Classification/House-Price.csv",header=TRUE)
str(df)
View(df)
summary(df)
boxplot(df$n_hot_rooms)
pairs(~df$Sold+df$rainfall)
barplot(table(df$bus_ter))

#Observations
#n_hot_rooms and rainfall has outliers
#n_hos_beds has missing values
#bus_ter is useless

uv <- 3*quantile(df$n_hot_rooms,0.99)
lv <- 0.3*quantile(df$rainfall,0.01)
df$n_hot_rooms[df$n_hot_rooms>uv] <-uv
df$rainfall[df$rainfall<lv] <- lv
summary(df)
boxplot(df$n_hot_rooms)
pairs(~df$Sold+df$rainfall)
which(is.na(df$n_hos_beds))
m <- mean(df$n_hos_beds,na.rm=TRUE)
df$n_hos_beds[is.na(df$n_hos_beds)] <- m
which(is.na(df$n_hos_beds))

summary(df$n_hos_beds)

df$avg_dist <- (df$dist1+df$dist2+df$dist3+df$dist4)/4
df <- df[,-6:-9]
df <- df [,-13]
install.packages("dummies")
df<-dummy.data.frame(df)
df <- df[,-8]
df <- df[,-13]

#Simple logistic model

glm_fit <- glm(Sold~price, data=df, family=binomial)
summary(glm_fit)

#multiple logistic model

glm_mul_fit <- glm(Sold~.,data=df,family=binomial)
summary(glm_mul_fit)

glm_probs <- predict(glm_fit, type="response")
glm_probs[1:10]
glm_predict <- rep("NO",506)
glm_predict[glm_probs>0.5] <-"YES"
View(glm_predict)
View(glm_probs)
table(glm_predict,df$Sold)

#Linear Discriminant Analysis

library("MASS")
lda_fit <- lda(Sold~.,data=df)
summary(lda_fit)
lda_fit
lda_predict <- predict(lda_fit,df)
lda_predict
lda_predict$posterior
lda_predict$class
lda_class <- lda_predict$class
lda_class
table(lda_class,df$Sold)
sum(lda_predict$posterior[,1]>0.8)

lda_class_b <- rep(0,506)
lda_class_b[lda_predict$posterior[,1]>0.8] <-1
table(lda_class_b,df$Sold)


#Quadratic discriminant analysis

qda_fit <- qda(Sold~.,data=df)
summary(qda_fit)
qda_fit
qda_predict <- predict(qda_fit,df)
qda_predict
qda_predict$posterior
qda_predict$class
qda_class <- qda_predict$class
qda_class
table(qda_class,df$Sold)
sum(lda_predict$posterior[,1]>0.8)

qda_class_b <- rep(0,506)
qda_class_b[qda_predict$posterior[,1]>0.8] <-1
table(qda_class_b,df$Sold)


#test-train split

set.seed(0)
split <- sample.split(df,SplitRatio = 0.8)
train_set <- subset(df,split==TRUE)
test_set <- subset(df,split==FALSE)

#Logistic Regression (Test-Train)
glm_train_fit <- glm(Sold~.,data=train_set,family=binomial)
summary(glm_train_fit)
View(test_set)
glm_test_probs <-predict(glm_train_fit,test_set,type="response")
glm_test_probs[1:10]
glm_test_predict <-rep("NO",120)
glm_test_predict[glm_test_probs>0.5] <- "YES"
View(glm_test_predict)
table(glm_test_predict,test_set$Sold)

#LDA (Test-Train)
lda_train_fit <- lda(Sold~.,data=train_set)
summary(lda_train_fit)
lda_train_fit
lda_test_predict <- predict(lda_train_fit,test_set)
View(lda_test_predict)
lda_test_predict$class
lda_test_predict$posterior
lda_test_class<-lda_test_predict$class
lda_test_class
table(lda_test_class,test_set$Sold)

lda_test_class_b <-rep(0,120)
lda_test_class_b[lda_test_predict$posterior[,1]<0.5]<-1
table(lda_test_class_b,test_set$Sold)

#KNN Classifier
train_x <- train_set[,-16]
test_x <- test_set[,-16]
train_y <- train_set$Sold
test_y <- test_set$Sold
k <-3
train_x_std <- scale(train_x)
test_x_std <- scale(test_x)
set.seed(0)
knn_predict <- knn(train_x_std,test_x_std,train_y,k=k)
knn_predict
table(knn_predict,test_y)
k<-1
knn_predict <- knn(train_x_std,test_x_std,train_y,k=k)
table(knn_predict,test_y)
