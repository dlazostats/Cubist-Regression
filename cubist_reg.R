# Bart Model
#------------
library(dplyr)
library(caret)
library(data.table)
library(mltools)
library(summarytools)

# Working directory
script_name <- 'cubist_reg.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

# Loading Data and preprocessing
df<-read.csv("salary.csv") 
df$rk<-as.factor(df$rk)
df.encoded <- one_hot(as.data.table(df)) %>% as.data.frame()

# EDA
df.encoded %>% descr(stats = "common")
featurePlot(x = df.encoded[,c("yr","yd")], 
            y = df.encoded$sl,  type = c("p", "smooth"),
            plot = "scatter", 
            layout = c(2, 1))

## Train/test
set.seed(1122)
trainIndex <- createDataPartition(df.encoded$sl, p = .8,list=F)
train <- df.encoded[trainIndex,]
test  <- df.encoded[-trainIndex,]

## controls
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,     
                           repeats = 5)

## ML models
#------------
### LM
lm_m<-train(sl~.,
            data=train,
            method="lm",
            trControl=fitControl)
summary(lm_m$finalModel)

### Cubist
cub_m<-train(sl~.,
             data=train,
             method="cubist",
             trControl=fitControl)
cub_m
summary(cub_m$finalModel)

# Error metrics
#--------------
# Train error
results <- resamples(list(LM=lm_m,
                          Cubist = cub_m)) 
dotplot(results,metric = "RMSE")
bwplot(results)
bwplot(results,metric = "RMSE")
bwplot(results,metric = "MAE")
bwplot(results,metric = "Rsquared")

# Test error 
model<-c("lm_m","cub_m")
lr<-list()
for(i in 1:2){
  pred<-predict(get(model[i]),newdata=test) 
  psam <-postResample(pred,test$sl) 
  lr[[i]]<-psam
}
dlr<-do.call("rbind",lr) %>% as.data.frame()
row.names(dlr)<-c("LM","Cubist")
dlr
