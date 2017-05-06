##Environment and import data
rm(list=objects())
graphics.off()
setwd(dir = "/Users/Kilian/Programmation/ENSTA_2A/R_project/MAE61/Spam-filter/")

##Library(used)
library(corrplot)
library(rfUtilities)
library(ROCR)
library(MASS)

##Function
averageWords <- function(dataFrame, spam, maxRow, maxCol)
{
  freq = 0
  freqAllWord = rep(0,maxCol)
  for(idCol in 1:maxCol){
    for (idRow in 1:maxRow){
      if(dataFrame[58][idRow] == spam){
        if(dataFrame[idCol][idRow] > 0)
          freq = freq + 1
      }
    }
    freqAllWord[idCol] = freq/maxRow
  }
  return (freqAllWord)
}
averageWordsSpam = averageWords(df_matrix, 1, 4601, 48)

##Question 1 :
df = read.table(file = 'spam.data')
dimDf = dim(df)
n = dim(df)[1]
names(df)[58] <- "Y"
corrplot(cor(df))
sum(df$Y)/4601 * 100 #39 % de spam
df_matrix = as.matrix(df)
#Variables très corrélées V32, V34 et V40
plot(df$V21, df$V23, col = df$Y + 2, pch = 20)
#Combinaison linéaire de 21 et 23 (qui sont très peu correlé)
#Si il y a la fois dans un mail les mots 23 et 21,
#alors il a de grandes chances d'être un spam

plot(df$V32[df$Y==0], df$V41[df$Y==0], col = df$Y + 2, pch = 20)

##Question 2 : 
# Y = \sigma X * \theta + eps

##Question 3 : 

set.seed(103)

# -------- Split into training and validation set--------

is_train = sample(c(TRUE,FALSE), n, rep = TRUE, prob = c(2/3,1/3))
X_train = df[is_train, - 58]
X_test = df[!is_train, - 58]
Y_train = df[is_train, 58]
Y_test = df[!is_train, 58]

#Question 4 : 
res.glm = glm(Y_train~., data = X_train, family = binomial)
summary(res.glm)
Y_proba_train.glm = predict(res.glm, X_train, type = "response")
Y_proba_test.glm = predict(res.glm, X_test, type = 'response')

##Question 5 : 
Y_pred_train = as.integer(Y_proba_train.glm > 0.5)
Y_pred_test = as.integer(Y_proba_test.glm > 0.5)

accuracy(Y_pred_train, Y_train) #93%
accuracy(Y_pred_test, Y_test) #92%

##Question 6 :

prediction(Y_pred_train, Y_train)
prediction(Y_pred_test, Y_test)

plot(performance(prediction(Y_proba_train.glm, Y_train), "tpr", "fpr"), col = 'black', lwd = 3)
plot(performance(prediction(Y_proba_train.glm, Y_train), "lift", "rpp"), col = 'black', lwd = 3)
performance(prediction(Y_proba_train.glm, Y_train), "auc")

plot(performance(prediction(Y_proba_test.glm, Y_test), "tpr", "fpr"), col = 'black', lwd = 3)
plot(performance(prediction(Y_proba_test.glm, Y_test), "lift", "rpp"), col = 'black', lwd = 3)
performance(prediction(Y_proba_test.glm, Y_test), "auc")

##Question 7 :
st1=stepAIC(res.glm)
#V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V15 + V16 + 
#  V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V26 + 
#  V27 + V28 + V29 + V33 + V35 + V38 + V39 + V41 + V42 + V43 + 
#  V44 + V45 + V46 + V48 + V49 + V52 + V53 + V54 + V56 + V57

##Question 8 : 

##Question 9 : 
#Il faut trouver le seuil qui autorise 5% de faux positifs. 
#On le détermine sur a courbe ROC.


#############Partie II
#library
library(glmnet)

##Question 2 : 
X_train_matrix = as.matrix(X_train)
res.glmnet = glmnet(X_train_matrix, Y_train, alpha = 0)
#10^-2 : presque regression non régularisé
#10^10 signifie : on ne feat plus

##Question 3 : 
res.glmnet.2 = glmnet(X_train_matrix, Y_train, lambda = 2, alpha = 0, family = "binomial")
Y_proba_2 = predict(res.glmnet.2, X_train_matrix, type = "response")
performance(prediction(Y_proba_2, Y_train),  "auc") #95%
performance(prediction(Y_proba_2, Y_train),  "acc") 

##Question 4 : 
plot(res.glmnet)


##Question 5 : 
