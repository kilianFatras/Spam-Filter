rm(list=objects())
graphics.off()
set.seed(103)

library(corrplot)
library(rfUtilities)
library(ROCR)
library(MASS)
library(glmnet)
library(caret) # for preprocessing (pca, normalization,...)

# ------------------------------------------------
# ------------------- Partie I -------------------
# ------------------------------------------------

# ================== Question 1 ==================

# _______________Lecture du dataset_______________

df = read.table(file = 'spam.data')
n = dim(df)[1]
m = dim(df)[2]
names(df)[58] <- "Y"

# ___________Affichage des correlations___________

corrplot(cor(df), tl.cex=0.6)

# corrplot partiel avec seulement Y
res.cor = cor(df)[58,-58, drop=FALSE]
corrplot(res.cor, tl.cex=0.7, tl.col = "black", cl.pos='n', 
         method="color", main="Correlation entre Y et les variables",
         mar=c(0,0,1,0))


# _____________Repartition des spams______________

sum(df$Y)/4601 * 100 # 39 % de spam

# ____Y en fonctions de plusieurs variables______

par(mfrow=c(1,2))
plot(df$V56[df$Y == 0], df$V57[df$Y == 0], col = df$Y[df$Y == 0] + 2, pch = 20,
     xlab="V56", ylab="V57", main="Répartition des non-spams sur V25 et V23")

plot(df$V56[df$Y == 1], df$V57[df$Y == 1], col = df$Y[df$Y == 1] + 2, pch = 20,
     xlab="V25", ylab="V23", main="Répartition des non-spams sur V25 et V23")

plot(df$V56, df$V55, col = df$Y + 2, pch = 20,
     xlab="V56", ylab="V55", main="Répartition des non-spams sur V25 et V23")

par(mfrow=c(1,2))
plot(df$V25[df$Y == 0], df$V23[df$Y == 0], col = df$Y[df$Y == 0] + 2, pch = 20,
     xlab="V25", ylab="V23", main="Répartition des non-spams sur V25 et V23")

plot(df$V25[df$Y == 1], df$V23[df$Y == 1], col = df$Y[df$Y == 1] + 2, pch = 20,
     xlab="V25", ylab="V23", main="Répartition des spams sur V25 et V23")

par(mfrow=c(1,2))
plot(df$V41[df$Y == 0], df$V32[df$Y == 0], col = df$Y[df$Y == 0] + 2, pch = 20,
     xlab="V41", ylab="V32", main="Répartition des non-spams sur V32 et V41")

plot(df$V41[df$Y == 1], df$V32[df$Y == 1], col = df$Y[df$Y == 1] + 2, pch = 20,
     xlab="V41", ylab="V32", main="Répartition des spams sur V32 et V41")

par(mfrow=c(1,1))
plot(df$V21, df$V23, col = df$Y + 2, pch = 20,
     xlab="V21", ylab="V23", main="Répartition des e-mails entre V21 et V23")


# __________Valeur moyenne des variables__________

averageFrequency <- function(df)
{
  nbVar = dim(df)[2] - 1
  freqInSpam = rep(0,nbVar)
  freqInNonSpam = rep(0,nbVar)
  
  for(iVar in 1:nbVar) {
    freqInSpam[iVar] = mean(df[,iVar][df$Y==1])
    freqInNonSpam[iVar] = mean(df[,iVar][df$Y==0])
  }
  print(freqInNonSpam)
  return (data.frame(freqInSpam, freqInNonSpam))
}

freq = averageFrequency(df[,c(-57,-56, -55, -54)])

plot(freq$freqInSpam, type="l", col="green",
     xlab="Variable", ylab="Valeur moyenne de la variable",
     main="Valeur moyenne des variables pour les spams et non-spams")
points(freq$freqInNonSpam, type="l", col="red")
legend(x=40, y=2.2, legend=c("spam", "non-spam"), fill=c("green","red"), col=c("green","red"))

# ================== Question 2 ==================

# Y = \sigma X * \theta + eps

# ================== Question 3 ================== 

is_train = sample(c(TRUE,FALSE), n, rep = TRUE, prob = c(2/3,1/3))
X_train = df[is_train, - 58]
X_test = df[!is_train, - 58]
Y_train = df[is_train, 58]
Y_test = df[!is_train, 58]

# ================== Question 4 ================== 

res.glm = glm(Y_train~., data = X_train, family = binomial)
summary(res.glm)
Y_proba_train.glm = predict(res.glm, X_train, type = "response")
Y_proba_test.glm = predict(res.glm, X_test, type = 'response')

# ================== Question 5 ================== 

Y_pred_train.glm = as.integer(Y_proba_train.glm > 0.5)
Y_pred_test.glm = as.integer(Y_proba_test.glm > 0.5)

accuracy(Y_pred_train.glm, Y_train)$PCC # 93.69%
accuracy(Y_pred_test.glm, Y_test)$PCC # 92.23%

# ================== Question 6 ================== 

prediction(Y_pred_train.glm, Y_train)
prediction(Y_pred_test.glm, Y_test)

# Lift curve on training set
plot(performance(prediction(Y_proba_train.glm, Y_train), "tpr", "rpp"), col = 'black', lwd = 3,
     main="Courbe Lift sur l'ensemble d'apprentissage")
n_spams_train = sum(Y_train)
n_train = sum(Y_train) + sum(!Y_train)
abline(h=1,col='red')
abline(0,1/(n_spams_train/n_train), col='red')
legend(x=0.4, y=0.2, legend=c("Régression logistique", "Modèle parfait"),
       fill=c("black", "red"))

# Lift curve on validation set
plot(performance(prediction(Y_proba_test.glm, Y_test), "tpr", "rpp"), col = 'black', lwd = 3,
     main="Courbe Lift sur l'ensemble de test")
n_spams_test = sum(Y_test)
n_test = sum(Y_test) + sum(!Y_test)
abline(0,1/(n_spams_train/n_train), col='red')
abline(h=1,col='red')
legend(x=0.4, y=0.2, legend=c("Régression logistique", "Modèle parfait"),
       fill=c("black", "red"))

# ROC curve on training set
plot(performance(prediction(Y_proba_train.glm, Y_train), "tpr", "fpr"), col = 'black', lwd = 3,
     main="Courbe ROC sur l'ensemble d'apprentissage")
points(x=c(0,0,1), y=c(0,1,1), type="l", col="red", lwd=2)
points(x=c(0,1), y=c(0,1), type="l", col="green", lwd=1, lty=2)
legend(x=0.4, y=0.2, legend=c("Régression logistique", "Modèle parfait", "Modèle aléatoire"),
       fill=c("black", "red","green"))

# ROC curve on validation set
plot(performance(prediction(Y_proba_test.glm, Y_test), "tpr", "fpr"), col = 'black', lwd = 1,
     main="Courbe ROC sur l'ensemble de test")
points(x=c(0,0,1), y=c(0,1,1), type="l", col="red", lwd=1)
points(x=c(0,1), y=c(0,1), type="l", col="green", lwd=1, lty=2)
#legend(x=0.4, y=0.2, legend=c("Régression logistique", "Modèle parfait", "Modèle aléatoire"),
#       fill=c("black", "red","green"))

# AUC on training set
performance(prediction(Y_proba_train.glm, Y_train), "auc") # 97.69%

# AUC on validation set
performance(prediction(Y_proba_test.glm, Y_test), "auc") # 97.43%

# ================== Question 7 ================== 

res.glm_select = glm(Y_train~V5+V7+V16+V17+V21+V23+V25+V45+V46+V53+V57, 
                     data = X_train, family = binomial)
Y_proba_train.glm_select = predict(res.glm_select, X_train, type = "response")
Y_proba_test.glm_select = predict(res.glm_select, X_test, type = 'response')

# ROC curve on validation set
plot(performance(prediction(Y_proba_test.glm_select, Y_test), "tpr", "fpr"), col = 'blue', lwd = 1,
     main="Courbe ROC sur l'ensemble de test", add=T)
points(x=c(0,0,1), y=c(0,1,1), type="l", col="red", lwd=2)
points(x=c(0,1), y=c(0,1), type="l", col="green", lwd=1, lty=2)
legend(x=0.5, y=0.25, legend=c("Modèle entier", "Sous-modèle", "Modèle parfait", "Modèle aléatoire"),
       fill=c("black", "blue", "red","green"))

# ================== Question 8 ================== 

st1 = stepAIC(res.glm)

# We keep these variables :
#  V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V15 + V16 + 
#  V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V26 + 
#  V27 + V28 + V29 + V33 + V35 + V38 + V39 + V41 + V42 + V43 + 
#  V44 + V45 + V46 + V48 + V49 + V52 + V53 + V54 + V56 + V57

res.glm_aic = glm(Y_train~V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V15 + V16 + 
                  V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V26 + 
                  V27 + V28 + V29 + V33 + V35 + V38 + V39 + V41 + V42 + V43 + 
                  V44 + V45 + V46 + V48 + V49 + V52 + V53 + V54 + V56 + V57, 
                  data = X_train, family = binomial)
Y_proba_train.glm_aic = predict(res.glm_aic, X_train, type = "response")
Y_proba_test.glm_aic = predict(res.glm_aic, X_test, type = 'response')

# ROC curve on validation set
plot(performance(prediction(Y_proba_test.glm_aic, Y_test), "tpr", "fpr"), col = 'blue', lwd = 1,
     main="Courbe ROC sur l'ensemble de test", add=T)
points(x=c(0,0,1), y=c(0,1,1), type="l", col="red", lwd=2)
points(x=c(0,1), y=c(0,1), type="l", col="green", lwd=1, lty=2)
legend(x=0.5, y=0.25, legend=c("Modèle entier", "Sous-modèle (AIC)", "Modèle parfait", "Modèle aléatoire"),
       fill=c("black", "blue", "red","green"))

performance(prediction(Y_proba_test.glm_aic, Y_test), "auc") # 97.54%

# ================== Question 9 ================== 

#Il faut trouver le seuil qui autorise 5% de faux positifs. 
#On le détermine sur a courbe ROC.

performance(prediction(Y_proba_test.glm_aic, Y_test), "tpr", "fpr")

# FPR = 0.05 => index = 558 => threshold = 0.432 et TPR = 0.9054
# On classifie donc comme spam tout Yi de proba >= 0.432

# ------------------------------------------------
# ------------------- Partie II ------------------
# ------------------------------------------------


# ================== Question 1 ================== 

# Le principe de la régression ridge est d'ajouter \lambda * norm2{\theta} dans
# la fonction de coût. Cela permet à la fois de rendre le problème identifiable 
# (car la solution dépend alors de l'inverse de XX' + lambda * I, matrice inversible
# si \lambda n'est pas vp), mais également d'éviter un sur-apprentissage, en garantissant
# un \theta pas trop grand.

# ================== Question 2 ================== 

X_train_matrix = as.matrix(X_train)
X_test_matrix = as.matrix(X_test)
res.glmnet = glmnet(X_train_matrix, Y_train, alpha = 0, 
                    lambda=c(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10))

# \lambda = 10^-2 : regression quasiment non régularisé => risque de sur-apprentissage

# \lambda = 10^10 signifie : on privilégie min \norm{\theta} à la minimisation de l'erreur
# La conséquence sera \theta = 0 et l'algorithme ne fittera plus

# ================== Question 3 ================== 

res.glmnet.2 = glmnet(X_train_matrix, Y_train, lambda = 2, alpha = 0, family = "binomial")
Y_proba_2 = predict(res.glmnet.2, X_test_matrix, type = "response")

performance(prediction(Y_proba_2, Y_test),  "auc") #93.6%
accuracy(as.integer(Y_proba_2 > 0.5), Y_test)

# ================== Question 4 ================== 

plot(res.glmnet)

# ================== Question 5 ================== 

cv.glmnet(X_train_matrix, Y_train, nfolds=10, alpha=0,
          lambda.min.ratio=1e-10, family="binomial")

# On teste de nouveau le modèle pour lambda.min

res.glmnet.min = glmnet(X_train_matrix, Y_train, lambda = 0.0004, alpha = 0.5, family = "binomial")
Y_proba_test.glm_min = predict(res.glmnet.min, X_test_matrix, type = "response")

performance(prediction(Y_proba_test.glm_min, Y_test),  "auc") # 96.9%
Y_proba_test.glm_minas.integer(Y_proba_min > 0.5)
accuracy(as.integer(Y_proba_min > 0.5), Y_test) # 89.44

# ------------------------------------------------
# ---------------- Autres methodes ---------------
# ------------------------------------------------

# ====================== PCA =====================

transf = preProcess(X_train, method=c("BoxCox", "center","scale", "pca"))
X_transf_train = predict(transf, X_train) # takes the PC that explains at least 95% of the variability in the data
X_transf_test = predict(transf, X_test)

res.glm_pca = glm(Y_train~., data = X_transf_train, family = binomial)
Y_proba_test.glm_pca = predict(res.glm_pca, X_transf_test, type = 'response')
Y_pred_test.glm_pca = as.integer(Y_proba_test.glm_pca >= 0.5)
accuracy(Y_pred_test.glm_pca, Y_test) # 92.03%

# ============ Methodes bayesiennes ==============

nbc = naiveBayes(Y_train~., data=X_train)
Y_proba_test.nbc = predict(nbc, X_test, type="raw")[,2]
Y_pred_test.nbc = as.integer(Y_proba_test.nbc>=0.5)
accuracy(Y_pred_test.nbc, Y_test)

# ================ Random Forest =================

library(randomForest)

rfc = randomForest(Y_train ~ ., data = X_train, na.action = na.roughfix, mtry=9, ntree=1000)
Y_proba_test.rfc = predict(rfc, X_test)
Y_pred_test.rfc = as.integer(Y_proba_test.rfc>=0.5)
accuracy(Y_pred_test.rfc, Y_test)

# ============= Comparison of methods ============

Y_pred_test.glm_aic = as.integer(Y_proba_test.glm_aic >= 0.5)
Y_pred_test.glm_select = as.integer(Y_proba_test.glm_select >= 0.5)
Y_pred_test.glm_lambda_2 = as.integer(Y_proba_2 >= 0.5)

acc_glm = accuracy(Y_pred_test.glm, Y_test)$PCC
acc_glm_select = accuracy(Y_pred_test.glm_select, Y_test)$PCC
acc_glm_aic = accuracy(Y_pred_test.glm_aic, Y_test)$PCC
acc_glm_lambda_2 = accuracy(Y_pred_test.glm_lambda_2, Y_test)$PCC
acc_glm_pca = accuracy(Y_pred_test.glm_pca, Y_test)$PCC
acc_nbc = accuracy(Y_pred_test.nbc, Y_test)$PCC
acc_rfc = accuracy(Y_pred_test.rfc, Y_test)$PCC

barplot(c(acc_glm,acc_glm_select,acc_glm_aic,acc_glm_lambda_2,acc_glm_pca,acc_nbc,acc_rfc), 
        names.arg=c("glm simple", "glm avec selection de variable", "glm avec stepAIC",
                    "ridge avec lambda=2", "ridge avec cv", "glm avec PCA", "Naive Bayes",
                    "Random Forest"))



