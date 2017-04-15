library('xgboost')
library('Metrics')
library('DiagrammeR')
library('corrplot')

train<-read.csv("numerai_training_data.csv")

train$target<-as.factor(train$target)

test<-read.csv("numerai_tournament_data.csv")


#outcomeName <- c('target')
#predictors <- names(train)[!names(train) %in% outcomeName]

#trainSet<-train[1:70000,]
#testSet<-train[70001:96320,]

a<-test$t_id
test$target<-'0'

test$t_id<-NULL


all<-rbind(train,test)

corrplot(cor(train[,-22]),method = "number")
Thershhold of 0.75

all$feature_1_8<-(all$feature1+all$feature8)/2

all$feature_2_14<-(all$feature2+all$feature14)/2
all$feature_2_19<-(all$feature2+all$feature19)/2

all$feature_3_5<-(all$feature3+all$feature5)/2

all$feature_4_16<-(all$feature4+all$feature16)/2
all$feature_4_20<-(all$feature4+all$feature20)/2

all$feature_5_13<-(all$feature5+all$feature13)/2

all$feature_7_8<-(all$feature7+all$feature8)/2

all$feature_9_10<-(all$feature9+all$feature10)/2

all$feature_10_12<-(all$feature10+all$feature12)/2

all$feature_11_15<-(all$feature11+all$feature15)/2

all$feature_15_18<-(all$feature15+all$feature18)/2

all$feature_17_21<-(all$feature17+all$feature21)/2

all$feature1<-NULL
all$feature2<-NULL
all$feature3<-NULL
all$feature4<-NULL
all$feature5<-NULL
all$feature7<-NULL
all$feature9<-NULL
all$feature10<-NULL
all$feature11<-NULL
all$feature15<-NULL
all$feature17<-NULL
all$feature8<-NULL
all$feature14<-NULL
all$feature19<-NULL
all$feature16<-NULL
all$feature20<-NULL
all$feature13<-NULL
all$feature8<-NULL
all$feature18<-NULL
all$feature21<-NULL

colnames(all)

corrplot(cor(all[,-3]),method = "number")


all$feature12<-NULL
all$feature_3_5<-NULL
all$feature_1_8<-NULL
all$feature_2_14<-NULL
all$feature_4_16<-NULL
all$feature_1_8<-NULL
all$feature_9_10<-NULL
all$feature_11_15<-NULL

colnames(all)

corrplot(cor(all[,-2]),method = "number")

train<-all[1:96320,]
test<-all[96321:231579,]

test$t_id<-a

outcomeName <- c('target')

predictors <- names(train)[!names(train) %in% outcomeName]
#predictors<-c("feature6","feature_4_20","feature_15_18","feature_7_8")

param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss",
              "eta" = 0.1, "max.depth" = 3)
bst.cv = xgb.cv(params=param, data = as.matrix(train[,predictors]), label = as.matrix(train[,outcomeName]), nfold = 10, nrounds = 1000,early.stop.round = 20)
plot(log(bst.cv$test.logloss.mean),type = "l")
bst <- xgboost(params=param,data = as.matrix(train[,predictors]), label = as.matrix(train[,outcomeName]), max.depth = 1, eta = 1, nround = 133,
               objective = "binary:logistic")

preds=predict(bst,as.matrix(test[,predictors]))


sub<-as.data.frame(test[,"t_id"])

colnames(sub)<-"t_id"
sub$probability<-preds
colnames(sub) <- c("t_id", "probability")
write.csv(sub,"sub4.csv",row.names = F)
#

xgb.importance(model=bst,feature_names = predictors)


library('caret')

cntr<-trainControl(method="repeatedcv",number = 10,repeats = 5)

model_glm<-train(train[,predictors],train[,outcomeName],method='glm',tun)


print(model_glm)


library('mRMRe')

dat<-train
dat$target<-NULL

ind <- sapply(dat, is.integer)
dat[ind] <- lapply(dat[ind], as.numeric)
dd <- mRMR.data(data = dat)
feats <- mRMR.classic(data = dd, target_indices = c(ncol(dat)), feature_count = 8)
variableImportance <-data.frame('importance'=feats@mi_matrix[nrow(feats@mi_matrix),])
variableImportance$feature <- rownames(variableImportance)
row.names(variableImportance) <- NULL
variableImportance <- na.omit(variableImportance)
variableImportance <- variableImportance[order(variableImportance$importance, decreasing=TRUE),]
print(variableImportance)

importance   feature
17  0.79285965 feature17
6   0.52596612  feature6
2   0.13514907  feature2
14  0.12824870 feature14
19  0.09985081 feature19
5   0.06383722  feature5
3   0.05889032  feature3
13  0.04430743 feature13
9  -0.05600996  feature9
12 -0.11199945 feature12
10 -0.13991052 feature10
11 -0.18006336 feature11
20 -0.20972844 feature20
16 -0.27594726 feature16
4  -0.28380861  feature4
15 -0.31850820 feature15
1  -0.32825453  feature1
18 -0.33439653 feature18
8  -0.45637957  feature8
7  -0.54275186  feature7


predictors<-c('feature17','feature6','feature7','feature8','feature18','feature1','feature15','feature4','feature16','feature20')

dtrain<-xgb.DMatrix(data=as.matrix(train[,predictors]),label=as.matrix(train[,outcomeName]))

param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss",
              "eta" = 0.1, "max.depth" = 1)
bst.cv = xgb.cv(params = param, dtrain, nfold = 10, nrounds = 1500,early.stop.round = 20)
plot(log(bst.cv$test.logloss.mean),type = "l")
bst <- xgboost(params = param,data = as.matrix(train[,predictors]), label = as.matrix(train[,outcomeName]), max.depth = 1, eta = 1, nround = 47,
               objective = "binary:logistic")

preds=predict(bst,as.matrix(test[,predictors]))


sub$t_id<-a
sub$probability<-preds
sub<-as.data.frame(sub)
colnames(sub) <- c("t_id", "probability")
write.csv(sub,"sub4.csv",row.names = F)
#

library(fscaret)
library(caret)


dput(names(train))

train<-train[,c("feature2", "feature4", "feature5", "feature8", "feature13", 
                "feature14", "feature19", "feature_6_20", "feature_7_15", 
                "feature_9_11", "feature_10_12", "feature_7_17", "feature_15_21", 
                "feature_16_1_21", "feature_3_17_18","target")]

set.seed(1234)
splitIndex <- createDataPartition(train$target, p = .75, list = FALSE, times = 1)
trainDF <- train[ splitIndex,]
testDF  <- train[-splitIndex,]

fsModels <- c("glm", "gbm", "treebag", "ridge", "lasso","deepBoost",'nnet') 

myFS<-fscaret(trainDF, testDF, myTimeLimit = 40, preprocessData=TRUE,
              Used.funcRegPred = fsModels, with.labels=TRUE,
              supress.output=FALSE)

myFS$VarImp$matrixVarImp.MSE

results <- myFS$VarImp$matrixVarImp.MSE
results$Input_no <- as.numeric(results$Input_no)
results <- results[c("SUM","SUM%","ImpGrad","Input_no")]
myFS$PPlabels$Input_no <-  as.numeric(rownames(myFS$PPlabels))
results <- merge(x=results, y=myFS$PPlabels, by="Input_no", all.x=T)
results <- results[c('Labels', 'SUM')]
results <- subset(results,results$SUM !=0)
results <- results[order(-results$SUM),]
print(results)



param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss",
              "eta" = 1, "max.depth" = 1)
bst.cv = xgb.cv(param=param, data = as.matrix(trainSet[,predictors]), label = as.matrix(trainSet[,outcomeName]), nfold = 10, nrounds = 200,early.stop.round = 10)
plot(log(bst.cv$test.logloss.mean),type = "l")
bst <- xgboost(data = as.matrix(train[,predictors]), label = as.matrix(train[,outcomeName]), max.depth = 1, eta = 1, nround = 16,
               objective = "binary:logistic")

preds=predict(bst,as.matrix(test[,predictors]))


sub$t_id<-test$t_id
sub$probability<-preds
sub<-as.data.frame(sub)
write.csv(sub,"sub1.csv",row.names = F)
#





# Get the feature real names
names <- dimnames(train[,predictors])
importance_matrix <- xgb.importance(feature_names =  predictors, model = bst)
xgb.plot.importance(importance_matrix[1:10])
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)





# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid,
#   using CV to evaluate
xgb_train = train(param=param,
                  data = as.matrix(train[,predictors]),
                  label = as.matrix(train[,outcomeName]),
                  trControl = xgb_trcontrol_1,
                  tuneGrid = xgb_grid_1,
                  method = "xgbLinear"
)


bst <- xgboost(data = as.matrix(train[,predictors]), label = as.matrix(train[,outcomeName]), 
               trControl = xgb_trcontrol_1,
               tuneGrid = xgb_grid_1,nround=200,
               objective = "binary:logistic")



# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) +
  geom_point() +
  theme_bw() 
scale_size_continuous(guide = "none")</code>
  
  
  
  
  
  
  
  
library(caret)

model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')

predictions<-predict(object = model_gbm,as.matrix(test[,predictors]),)

sub$t_id<-test$t_id
sub$probability<-0.5
sub<-as.data.frame(sub)
write.csv(sub,"sub.csv",row.names = F)


