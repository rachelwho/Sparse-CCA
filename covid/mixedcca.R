library(mixedCCA)
setwd(here::here())
X = read.csv('X_covid.csv', header = FALSE)
Y = read.csv('Y_covid.csv', header = FALSE)
e_sig = estimateR_mixed(X, Y,  type1 = "continuous", type2 = "trunc")
Rhat = e_sig[[5]]
write.csv(e_sig[[5]],"Rhat.csv")

mixedCCAresult1 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", BICtype = 1, KendallR = Rhat)
mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", BICtype = 2, KendallR = Rhat)
#mixedCCAresult <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", BICtype = 2)
mixedCCAresult = mixedCCAresult1
w1 = (mixedCCAresult$w1 / norm(mixedCCAresult$w1, "2"))[,1]
# 9 16
w2 = (mixedCCAresult$w2 / norm(mixedCCAresult$w2, "2"))[,1]
# 58  79 134 489
mixedCCAresult$cancor
# 0.9036071

mixedCCAresult = mixedCCAresult2
w1 = (mixedCCAresult$w1 / norm(mixedCCAresult$w1, "2"))[,1]
# 1  3  5  8  9 11 12 13 16
w2 = (mixedCCAresult$w2 / norm(mixedCCAresult$w2, "2"))[,1]
# 58  79 134 489
mixedCCAresult$cancor
# 0.9267141
