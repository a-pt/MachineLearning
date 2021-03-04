movie <- read.csv("D:/Online courses/ML(Rstudio)/resources/Complete ML in R/4. Support Vector Machines/Movie_classification.csv")
summary(movie)
movie$Time_taken[is.na(movie$Time_taken)] <- mean(movie$Time_taken,na.rm=TRUE)
install.packages("caTools")
library(caTools)
set.seed(0)
split <- sample.split(movie,SplitRatio = 0.8)
train_set <- subset(movie,split==TRUE)
test_set <- subset(movie,split==FALSE)

#Classification
test_set$Start_Tech_Oscar <- as.factor(test_set$Start_Tech_Oscar)
train_set$Start_Tech_Oscar <- as.factor(train_set$Start_Tech_Oscar)
install.packages("e1071")
library(e1071)

#Linear Kernel
svmfit <- svm(Start_Tech_Oscar~.,data=train_set,kernel="linear",cost=1,scale=TRUE)
summary(svmfit)
pred <- predict(svmfit,test_set)
pred
table(predict=pred,truth=test_set$Start_Tech_Oscar)
66/107
svmfit$index

#cost tuning(linear)
set.seed(0)
tune_out <- tune(svm,Start_Tech_Oscar~.,data=train_set,kernel="linear",range=list(cost=c(0.1,0.5,0.8,1,1.5,2,8,10),cross=20))
bestmodelsvm <- tune_out$best.model
summary(bestmodelsvm)
pred_best <-predict(bestmodelsvm,test_set)
table(predict=pred_best,truth=test_set$Start_Tech_Oscar)
61/107


#Polynimial Kernel

svmfitp <- svm(Start_Tech_Oscar~.,data=train_set,kernel="polynomial",cost=1,degree=2)
summary(svmfitp)
pred_p <-predict(svmfitp,test_set)
table(predict=pred_p,truth=test_set$Start_Tech_Oscar)
64/107

tune_outp <-tune(svm,Start_Tech_Oscar~.,data=train_set,cross=4,kernel="polynomial",ranges=list(cost=c(0.001,0.1,1,5,10),degree=c(0.5,1,2,3,5)))
bestmodelsvmp <- tune_outp$best.model
summary(bestmodelsvmp)
pred_best_p <-predict(bestmodelsvmp,test_set)
table(predict=pred_best_p,truth=test_set$Start_Tech_Oscar)
65/107

#Radial Kernel
svmfitr <- svm(Start_Tech_Oscar~.,data=train_set,kernel="radial",gamma=1,cost=1)
summary(svmfitr)
pred_r <-predict(svmfitr,test_set)
table(predict=pred_r,truth=test_set$Start_Tech_Oscar)
56/107

tune_outr <- tune(svm,Start_Tech_Oscar~.,data=train_set,kernel="radial",ranges=list(cost=c(0.001,0.1,1,10,100,1000),gamma=c(0.01,0.1,0.5,1,3,10,50)),cross=4)
summary(tune_outr)
bestmodelsvmr <- tune_outr$best.model
summary(bestmodelsvmr)
pred_best_r <- predict(bestmodelsvmr,test_set)
table(predict=pred_best_r,truth=test_set$Start_Tech_Oscar)
58/107


#SVM For Regression

df <- read.csv("D:/Online courses/ML(Rstudio)/resources/Complete ML in R/4. Support Vector Machines/Movie_regression.csv",header=TRUE)
summary(df)
df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken,na.rm=TRUE)

#install.packages("caTools")
library(caTools)
set.seed(0)
split <- sample.split(df,SplitRatio = 0.8)
train_set_r <- subset(df,split==TRUE)
test_set_r <- subset(df,split==FALSE)

#Simple LinearKernel

#install.package("e1071")
library(e1071)
svmfitreg <-svm(Collection~.,data=train_set_r,kernel="linear",cost=0.01,scale=TRUE)
summary(svmfitreg)

pred_reg <- predict(svmfitreg,test_set_r)
pred_reg
mse2_svm_reg <- mean((pred_reg-test_set_r$Collection)^2)

#Simple polynomial Kernel
svmfitregp <-svm(Collection~.,data=train_set_r,kernel="polynomial",cost=0.01,scale=TRUE,degree=2)
summary(svmfitregp)

pred_reg_p <- predict(svmfitregp,test_set_r)
pred_reg_p
mse2_svm_reg_p <- mean((pred_reg_p-test_set_r$Collection)^2)

#simple Radial Kernel
svmfitregr <-svm(Collection~.,data=train_set_r,kernel="radial",cost=0.01,scale=TRUE,gamma=1)
summary(svmfitregr)

pred_reg_r <- predict(svmfitregr,test_set_r)
pred_reg_r
mse2_svm_reg_r <- mean((pred_reg_r-test_set_r$Collection)^2)
