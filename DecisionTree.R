#Decision Regression Tree

movie <- read.csv("D:/Online courses/ML(Rstudio)/resources/Complete ML in R/3. Decision Trees/Movie_regression.csv",header=TRUE)
View(movie)

#Prepocessing

summary(movie)
plot(movie$Collection,movie$Marketing.expense)
mean <- mean(movie$Time_taken,na.rm=TRUE)
movie$Time_taken[is.na(movie$Time_taken)] <- mean

#Test Train split

install.packages("caTools")
library(caTools)
set.seed(0)
split <- sample.split(movie,SplitRatio = 0.8)
train_set <- subset(movie,split==TRUE)
test_set <- subset(movie,split==FALSE)


#install packages

install.packages('rpart')
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

?rpart

regtree <- rpart(formula=Collection~.,data=train_set,control=rpart.control(maxdepth=3))
View(regtree)

rpart.plot(regtree,box.palette="RdBu",digits=-3)

#prediction
test_set$predict <- predict(regtree,test_set,type="vector")

#performance
mse2 <- mean((test_set$predict-test_set$Collection)^2)

#pruning
fulltree <- rpart(formula=Collection~.,data=train_set,control=rpart.control(cp=0))
rpart.plot(fulltree,box.palette="RdBu",digits=-3)

printcp(fulltree)
plotcp(regtree)
index<-which.min(regtree$cptable[,"xerror"])
min_cp <-regtree$cptable[index,"CP"]
prunedtree <- prune(fulltree,cp=min_cp)
rpart.plot(prunedtree,box.palette="RdBu",digits=-3)

#comparison(full,purned,reg)

test_set$full_predict <- predict(fulltree,test_set,type="vector")
full_mse2 <- mean((test_set$full_predict-test_set$Collection)^2)

test_set$pur_predict <- predict(prunedtree,test_set,type="vector")
prune_mse2 <- mean((test_set$pur_predict-test_set$Collection)^2)



#Classification Decision Tree

df <- read.csv("D:/Online courses/ML(Rstudio)/resources/Complete ML in R/3. Decision Trees/Movie_classification.csv",header=TRUE)
View(df)

summary(df)
df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken,na.rm=TRUE)
summary(df)

install.packages("caTools")
library(caTools)
set.seed(0)
split_c <-sample.split(df,SplitRatio = 0.8)
train_set_c <- subset(df,split_c==TRUE)
test_set_c <-subset(df,split_c==FALSE)

#install.packages("rpart")
#install.packages("rpart.plot")

classtree<- rpart(formula = Start_Tech_Oscar~.,data=train_set_c,method = "class",control=rpart.control(maxdepth=3))
rpart.plot(classtree,box.palette = "RdBu",digits=-3)

test_set_c$predict <- predict(classtree,test_set_c,type="class")
View(test_set_c)
table(test_set_c$predict,test_set_c$Start_Tech_Oscar)
64/107

#Bagging (Regression)
install.packages("randomForest")
library(randomForest)
#Do test train spilt,seed set
bagging <- randomForest(formula=Collection~.,data=train_set,mtry=17)
summary(bagging)

test_set$bagging_pred <- predict(bagging,test_set)
mse2_bagging<-mean((test_set$bagging_pred-test_set$Collection)^2)


#Random Forest (Regression)

#install.packages("randomForest")
#library(randomForest)

randomforest <- randomForest(Collection~.,train_set,ntree=500)
test_set$rfpredict <- predict(randomforest,test_set)
mse2_rf <- mean((test_set$rfpredict-test_set$Collection)^2)

#Gradient Boost
install.packages("gbm")
library(gbm)

#Preprocessing 
#install.packages("dummies")
#library(dummies)
df2 <- movie
df2<-dummy.data.frame(df2)
df2 <- df2[,-12]
df2 <- df2[,-15]
set.seed(0)
split <- sample.split(df2,SplitRatio = 0.8)
train_set_gb <- subset(df2,split==TRUE)
test_set_gb <- subset(df2,split==FALSE)
gboosting <- gbm(Collection~.,data=train_set_gb,distribution = "gaussian", n.trees = 5000,interaction.depth = 4,shrinkage = 0.2,verbose = F)
summary(gboosting)
test_set_gb$predict <- predict(gboosting,test_set_gb,n.trees = 5000)
gboost_mse2 <- mean((test_set_gb$predict-test_set_gb$Collection)^2)


#AdaBoost

install.packages("adabag")
library(adabag)
train_set_c$Start_Tech_Oscar<-as.factor(train_set_c$Start_Tech_Oscar)
adaboost <- boosting(Start_Tech_Oscar~.,train_set_c,boos=TRUE)
ada_predict <- predict(adaboost,test_set_c)
table(ada_predict$class,test_set_c$Start_Tech_Oscar)

t1<-adaboost$trees[[1]]
plot(t1)
text(t1,pretty=100)

#XG Boost

install.packages("xgboost")
library(xgboost)

#Dmatrix format

train_y <- train_set_c$Start_Tech_Oscar=='1'
train_x <- model.matrix(Start_Tech_Oscar~.,-1,data=train_set_c)
train_x <- train_x[,-1]

test_y <- test_set_c$Start_Tech_Oscar=='1'
test_x <- model.matrix(Start_Tech_Oscar~.,-1,data=test_set_c)
test_x <- test_x[,-22]
test_x <- test_x[,-1]
Xmatrix_train <- xgb.DMatrix(data=train_x,label=train_y)
Xmatrix_test <- xgb.DMatrix(data=test_x,label=test_y)

xgboosting <- xgboost(data=Xmatrix_train,nrounds=50,objective="multi:softmax",eta=0.3,num_class=2,max_depth=10)
xgpred <- predict(xgboosting,Xmatrix_test)
table(xgpred,test_y)
68/107
