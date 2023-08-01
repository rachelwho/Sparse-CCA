library(dplyr)
library(mixedCCA)
setwd("/projectnb/labci/Qiuyun/SCCA/v2/mixdata")
# setwd("/Users/apple/Desktop/Dropbox/Sparse CCA/simulation/v2/continuous")

path = "100-100_180_0.7_0.8/"
tol = 0.00001
maxiter = 5000
record_trace = FALSE
Init = TRUE
num_run = 100
dir.create(file.path(getwd(), paste0("data/", path)), showWarnings = TRUE)
dir.create(file.path(getwd(), paste0("result/", path)), showWarnings = TRUE)
dir.create(file.path(getwd(), paste0("result/", path,Init, 'mixedcca')), showWarnings = TRUE)
# f = function(x, power=1){
#   return(abs(x)^{power}*sign(x))
# }
# f_inv = function(x, power=1){
#   return(abs(x)^{1/power}*sign(x))
# }

SigX = as.matrix(read.csv(paste0("../continuous/data/", path,"SigX.csv"), header = FALSE))
SigY = as.matrix(read.csv(paste0("../continuous/data/", path,"SigY.csv"), header = FALSE))
vx = read.csv(paste0("../continuous/data/", path,"vx.csv"), header = FALSE)[,1]
vy = read.csv(paste0("../continuous/data/", path,"vy.csv"), header = FALSE)[,1]
vx = vx / norm(vx, "2")
vy = vy / norm(vy, "2")
# Sig12 = 0.9*SigX%*%vx%*%(t(vy)%*%SigY) / sqrt((vx%*%SigX%*%vx)[1]) / sqrt((vy%*%SigY%*%vy)[1])

p1 = length(vx)
p2 = length(vy)
p = p1+p2

ind = 14
res = rep(list(matrix(0, nrow = num_run, ncol = ind)), 3)
W_m = rep(list(matrix(0, nrow = num_run, ncol = 2*p)), 3)
iter = rep(list(matrix(0,  nrow = num_run, ncol = 2)), 3)


cal_error = function(w, v){
  w = w / norm(w, "2")
  return(min(norm(v- w, "2")^2, norm(v+ w, "2")^2))
}
cal_tpr = function(w, v){
  return(sum(w[v!=0] != 0)/3)
}
cal_tnr = function(w, v){
  return(sum(w[v==0] == 0)/(p2-3))
}
### Default initial starting point


# recordtrace = function(mixedCCAresult, k, index = 1){
#   error = matrix(nrow = 7, ncol = maxiter+1)
#   iter_mix = dim(mixedCCAresult$fitresult$w1trace)[2]
#   error[1,] = apply(mixedCCAresult$fitresult$w1trace, 2, cal_error,v = vx)[1:(maxiter+1)]
#   error[2,] = apply(mixedCCAresult$fitresult$w2trace, 2, cal_error, v = vy)[1:(maxiter+1)]
#   error[3,] = apply(mixedCCAresult$fitresult$w1trace, 2, cal_tpr, v = vx)[1:(maxiter+1)]
#   error[4,] = apply(mixedCCAresult$fitresult$w2trace, 2, cal_tpr, v = vy)[1:(maxiter+1)]
#   error[5,] = apply(mixedCCAresult$fitresult$w1trace, 2, cal_tnr, v = vx)[1:(maxiter+1)]
#   error[6,] = apply(mixedCCAresult$fitresult$w2trace, 2, cal_tnr, v = vy)[1:(maxiter+1)]
#   error[7, 1:iter_mix] = seq(0, res[k, 12+index], length.out = iter_mix)
#   write.table(error[,1:iter_mix],paste0("result/", path, Init, "mixedcca/error-time",index, "_",k-1,".csv"), row.names = FALSE, col.names = FALSE)
# }


for(k in 1:num_run){
  i = 1
  data = read.csv( paste0("../continuous/data/", path,"data",k-1,".csv") ,header = FALSE)
  for(C in c(-2, -1, 0)){
    Rhat <- read.csv(paste0("data/", path,"R_exact_PD",k-1, "_",C,".csv"), header = TRUE, row.names = 1)
    X = data[,1:p1]
    Y = data[,(p1+1):(p1+p2)]
    Y = Y-C
    Y[Y<0] = 0
    w = as.numeric(read.csv(paste0("data/", path,"/minit_exact_SPD",k-1, "_",C,".csv"), header = FALSE)[,1])

    time1 = Sys.time()
    if(Init){
      mixedCCAresult1 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", w1init = w[1:p1],
                                  w2init = w[(p1+1):p], KendallR = Rhat,  tol = tol, BICtype = 1, maxiter = maxiter, trace = T)
    }else{
      mixedCCAresult1 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc",
                                  tol = tol, KendallR = Rhat, BICtype = 1, maxiter = maxiter, trace = F)
      
    }
    print(Sys.time() - time1)
    ## computation time
    res[[i]][k, 13] = as.numeric (Sys.time() - time1, units = "secs")

    time2 = Sys.time()
    if(Init){
      mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", w1init = w[1:p1],
                                  w2init = w[(p1+1):p], KendallR = Rhat, tol = tol, BICtype = 2, maxiter = maxiter, trace = T)
    }else{
      mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc",
                                  tol = tol, KendallR = Rhat, BICtype = 2, maxiter = maxiter, trace = F)
      
    }  
    res[[i]][k, 14] = as.numeric (Sys.time() - time2, units = "secs")
    
    mixedCCAresult = mixedCCAresult1
    iter[[i]][k, 1] = dim(mixedCCAresult$fitresult$w1trace)[2]
    w1 = (mixedCCAresult$w1 / norm(mixedCCAresult$w1, "2"))[,1]
    w2 = (mixedCCAresult$w2 / norm(mixedCCAresult$w2, "2"))[,1]
    W_m[[i]][k,1:p1] = w1
    W_m[[i]][k,(p1+1):p] = w2
    ## squared l2 distance
    res[[i]][k, 1] = cal_error(w1, vx)
    res[[i]][k, 2] = cal_error(w2, vy)
    ## TPR & TNR
    res[[i]][k, 3] = sum(w1[vx != 0] != 0)/3
    res[[i]][k, 4] = sum(w1[vy != 0] != 0)/3
    res[[i]][k, 5] = sum(w1[vx == 0] == 0) / (p1-3)
    res[[i]][k, 6] = sum(w2[vy == 0]  == 0)/ (p2-3)

    ## BIC2
    mixedCCAresult = mixedCCAresult2
    iter[[i]][k, 2] = dim(mixedCCAresult$fitresult$w1trace)[2]
    w1 = (mixedCCAresult$w1 / norm(mixedCCAresult$w1, "2"))[,1]
    w2 = (mixedCCAresult$w2 / norm(mixedCCAresult$w2, "2"))[,1]
    W_m[[i]][k,(p+1):(p1+p)] = w1
    W_m[[i]][k,(p1+1+p):(2*p)] = w2
    ## squared l2 distance
    res[[i]][k, 7] = min(norm(vx- w1, "2")^2, norm(vx+ w1, "2")^2)
    res[[i]][k, 8] = min(norm(vy- w2, "2")^2, norm(vy+ w2, "2")^2)
    ## TPR & TNR
    res[[i]][k, 9] = sum(w1[vx != 0] != 0)/3
    res[[i]][k, 10] = sum(w2[vy != 0] != 0)/3
    res[[i]][k, 11] = sum(w1[vx == 0] == 0) / (p1-3)
    res[[i]][k, 12] = sum(w2[vy == 0] == 0)/ (p2-3)
    
    cat(k,C,'\n', res[[i]][k,], '\n')
    i = i+1
  }
}

write.table(res,paste0("result/", path,Init, "mixedcca/error_exact_PD.csv"), row.names = FALSE, col.names = FALSE)
write.table(W_m,paste0("result/", path,Init, "mixedcca/theta_exact_PD.csv"), row.names = FALSE, col.names = FALSE)
write.table(iter,paste0("result/", path,Init, "mixedcca/iter_exact_PD.csv"), row.names = FALSE, col.names = FALSE)
# W_m= read.table("result/mix_theta.csv")
path = "100-100_0.7_0.8/"
ind = 14
rrr = read.table(paste0("result/", path,Init, "mixedcca/error_exact_PD.csv"))
rr = list(rrr[,1:ind], rrr[(ind + 1):(2*ind)], rrr[(2*ind + 1): (3*ind)])
rr = res
me = matrix(0, 3, ind)
std = matrix(0, 3, ind)
## squared l2 distance
for (i in 1:3) {
  me[i,] = round(apply(rr[[i]], 2, mean),2)
  std[i,] = round(apply(rr[[i]], 2, sd) ,2)
  for (j in c(1,2,7,8)) {
    cat("&", me[i,j], "(", std[i,j], ")" ) 
    
  }
  cat('\n')
}
## TPR
for (i in 1:3) {
  for (j in c(3,4,9,10)) {
    cat("&", me[i,j], "(", std[i,j], ")" ) 
  }
  cat('\n')
}
## TNR
for (i in 1:3) {
  for (j in c(5,6,11,12)) {
    cat("&", me[i,j], "(", std[i,j], ")" ) 
  }
  cat('\n')
}
## computation
for (i in 1:3) {
  for (j in c(13,14)) {
    cat("&", me[i,j], "(", std[i,j], ")" ) 
  }
  cat('\n')
}
## iteration


