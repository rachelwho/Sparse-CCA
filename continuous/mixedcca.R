library(dplyr)
library(mixedCCA)
setwd("/projectnb/labci/Qiuyun/SCCA/v2/continuous")
# setwd("/Users/apple/Desktop/Dropbox/Sparse CCA/simulation/v2/continuous")

path = "250-250_0.7_0.8/"
tol = 0.001
maxiter = 2000
record_trace = TRUE
Init = FALSE
dir.create(file.path(getwd(), paste0("result/", path)), showWarnings = TRUE)
dir.create(file.path(getwd(), paste0("result/", path,Init, 'mixedcca')), showWarnings = TRUE)

SigX = as.matrix(read.csv(paste0("data/", path,"SigX.csv"), header = FALSE))
SigY = as.matrix(read.csv(paste0("data/", path,"SigY.csv"), header = FALSE))
vx = read.csv(paste0("data/", path,"vx.csv"), header = FALSE)[,1]
vy = read.csv(paste0("data/", path,"vy.csv"), header = FALSE)[,1]
vx = vx / norm(vx, "2")
vy = vy / norm(vy, "2")
# Sig12 = 0.9*SigX%*%vx%*%(t(vy)%*%SigY) / sqrt((vx%*%SigX%*%vx)[1]) / sqrt((vy%*%SigY%*%vy)[1])

p1 = length(vx)
p2 = length(vy)
p = p1+p2
num_run = 100
ind = 14
res = matrix(0, nrow = num_run, ncol = ind)
W_m = matrix(0, nrow = num_run, ncol = 2*p)

## generate Rhat
# for(k in 1:num_run){
#   i = 1
#   data = read.csv( paste0("data/data",k-1,".csv") ,header = FALSE)
#   for(power in c(1)){
#     for(C in c(-2, -1, 0)){
      # X = data[,1:p1]
      # Y = data[,(p1+1):(p1+p2)]
      # time1 = Sys.time()
      # e_sig2 = estimateR_mixed(X, Y,  type1 = "continuous", type2 = "continuous")
      # write.csv(e_sig2[[5]],paste0("data/R",k-1, "_",C,".csv"))
      # time2 = Sys.time()
      # print(time2 - time1)
#       i = i+1
#     }
#   }
# }
### Default initial starting point
InitF <- function(X1, X2, type1, type2,
                  lamseq1 = NULL, lamseq2 = NULL, nlamseq = 20, lam.eps = 1e-02,
                  w1init = NULL, w2init = NULL, BICtype,
                  KendallR = NULL,
                  maxiter = 100, tol = 1e-2, trace = FALSE, lassoverbose = FALSE){
  n <- nrow(X1)
  p1 <- ncol(X1); p2 <- ncol(X2);
  p <- p1 + p2
  
  ### Compute Kendall tau if there is no input.
  if(is.null(KendallR)){
    R <- estimateR_mixed(X1, X2, type1 = type1, type2 = type2)$R
  } else {
    R <- KendallR; rm(KendallR)
  }
  
  R1 <- as.matrix(R[1:p1, 1:p1])
  R2 <- as.matrix(R[(p1+1):p, (p1+1):p])
  R12 <- as.matrix(R[1:p1, (p1+1):p])
  
  ### Default initial starting point
  if (is.null(w1init) | is.null(w2init)){
    RCCA <- myrcc(R1 = R1, R2 = R2, R12 = R12, lambda1 = 0.25, lambda2 = 0.25)
    if (is.null(w1init)){
      w1init <- as.matrix(RCCA$w1, ncol=1)
    }
    if (is.null(w2init)){
      w2init <- as.matrix(RCCA$w2, ncol=1)
    }
  }
  
  # standardize initial starting points - for both lambda seq generation (lambdaseq_generate) and cca algorithm (find_w12bic)
  
  # write.table(c(w1init, w2init), sep = ',', file = paste0("data/", path,"/minit",k-1,".csv"), col.names = FALSE, row.names = FALSE)
}


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
recordtrace = function(mixedCCAresult, k, index = 1){
  error = matrix(nrow = 7, ncol = maxiter+1)
  iter_mix = dim(mixedCCAresult$fitresult$w1trace)[2]
  error[1,] = apply(mixedCCAresult$fitresult$w1trace, 2, cal_error,v = vx)[1:(maxiter+1)]
  error[2,] = apply(mixedCCAresult$fitresult$w2trace, 2, cal_error, v = vy)[1:(maxiter+1)]
  error[3,] = apply(mixedCCAresult$fitresult$w1trace, 2, cal_tpr, v = vx)[1:(maxiter+1)]
  error[4,] = apply(mixedCCAresult$fitresult$w2trace, 2, cal_tpr, v = vy)[1:(maxiter+1)]
  error[5,] = apply(mixedCCAresult$fitresult$w1trace, 2, cal_tnr, v = vx)[1:(maxiter+1)]
  error[6,] = apply(mixedCCAresult$fitresult$w2trace, 2, cal_tnr, v = vy)[1:(maxiter+1)]
  error[7, 1:iter_mix] = seq(0, res[k, 12+index], length.out = iter_mix)
  write.table(error[,1:iter_mix],paste0("result/", path, Init, "mixedcca/error-time",index, "_",k-1,".csv"), row.names = FALSE, col.names = FALSE)
}

for(k in 1:num_run){
  data = read.csv( paste0("data/", path,"/data",k-1,".csv") ,header = FALSE)
  n = length(data)
  X = data[,1:p1]
  Y = data[,(p1+1):(p1+p2)]
  e_sig2 = estimateR_mixed(X, Y,  type1 = "continuous", type2 = "continuous")
  write.csv(e_sig2[[5]],paste0("data/", path,"/Rhat",k-1,".csv"))
  if(Init){
    w = as.numeric(read.csv(paste0("data/", path,"/init",k-1,".csv"), header = FALSE)[,1])
  }else{
    InitF(X, Y, type1 = "continuous", type2 = "continuous",
          tol = tol, BICtype = 1, maxiter = maxiter, trace = T)
  }
  
  time1 = Sys.time()
  if(Init){
    mixedCCAresult1 <- mixedCCA(X, Y, type1 = "continuous", type2 = "continuous", w1init = w[1:p1],
                                w2init = w[(p1+1):p], tol = tol, BICtype = 1, maxiter = maxiter, trace = T)
  }else{
    mixedCCAresult1 <- mixedCCA(X, Y, type1 = "continuous", type2 = "continuous",
                                tol = tol, BICtype = 1, maxiter = maxiter, trace = T)
    
  }
  print(Sys.time() - time1)
  ## computation time
  res[k, 13] = as.numeric (Sys.time() - time1, units = "secs")
  time2 = Sys.time()
  if(Init){
    mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "continuous", w1init = w[1:p1],
                                w2init = w[(p1+1):p], tol = tol, BICtype = 2, maxiter = maxiter, trace = T)
  }else{
    mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "continuous",
                                tol = tol, BICtype = 2, maxiter = maxiter, trace = T)
    
  }  
  res[k, 14] = as.numeric (Sys.time() - time2, units = "secs")
  #mixedCCAresult <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", BICtype = 2)
  if(record_trace){
    recordtrace(mixedCCAresult1,k, 1)
    ## record trace error
    recordtrace(mixedCCAresult2,k, 2)
  }
  mixedCCAresult = mixedCCAresult1
  ## record trace error
  
  
  
  w1 = (mixedCCAresult$w1 / norm(mixedCCAresult$w1, "2"))[,1]
  w2 = (mixedCCAresult$w2 / norm(mixedCCAresult$w2, "2"))[,1]
  W_m[k,1:p1] = w1
  W_m[k,(p1+1):p] = w2
  ## squared l2 distance
  res[k, 1] = cal_error(w1, vx)
  res[k, 2] = cal_error(w2, vy)
  ## TPR & TNR
  res[k, 3] = sum(w1[vx != 0] != 0)/3
  res[k, 4] = sum(w1[vy != 0] != 0)/3
  res[k, 5] = sum(w1[vx == 0] == 0) / (p1-3)
  res[k, 6] = sum(w2[vy == 0]  == 0)/ (p2-3)
  ## expected out of sample correlation
  # res[[i]][k, 3] = w1%*%Sig12%*%w2 / sqrt(w1%*%Sig%*%w1) / sqrt(w2%*%Sig%*%w2)
  ## predictive loss
  # res[[i]][k, 4] = 1 - abs(w1%*%Sig%*%vx/sqrt(w1%*%Sig%*%w1))
  # res[[i]][k, 5] = 1 - abs(w2%*%Sig%*%vx/sqrt(w1%*%Sig%*%w2))
  
  ## BIC2
  mixedCCAresult = mixedCCAresult2
  
  w1 = (mixedCCAresult$w1 / norm(mixedCCAresult$w1, "2"))[,1]
  w2 = (mixedCCAresult$w2 / norm(mixedCCAresult$w2, "2"))[,1]
  W_m[k,(p+1):(p1+p)] = w1
  W_m[k,(p1+1+p):(2*p)] = w2
  ## squared l2 distance
  res[k, 7] = min(norm(vx- w1, "2")^2, norm(vx+ w1, "2")^2)
  res[k, 8] = min(norm(vy- w2, "2")^2, norm(vy+ w2, "2")^2)
  ## TPR & TNR
  res[k, 9] = sum(w1[vx != 0] != 0)/3
  res[k, 10] = sum(w2[vy != 0] != 0)/3
  res[k, 11] = sum(w1[vx == 0] == 0) / (p1-3)
  res[k, 12] = sum(w2[vy == 0] == 0)/ (p2-3)
  
  cat(k,'\n', res[k,], '\n')
  
}
write.table(res,paste0("result/", path,Init, "mixedcca/error.csv"), row.names = FALSE, col.names = FALSE)
write.table(W_m,paste0("result/", path,Init, "mixedcca/theta.csv"), row.names = FALSE, col.names = FALSE)

# W_m= read.table("result/mix_theta.csv")

### read results
rr = res
path = "100-100_0.7_0.8/"
rr = read.table(paste0("result/", path, "FALSEmixedcca/error.csv"))
## squared l2 distance
me= round(apply(rr, 2, mean),2)
std = round(apply(rr, 2, sd) ,2)
for (j in c(1,2,7,8)) {
  cat("&", me[j], "(", std[j], ")" ) 
}
## TPR
for (j in c(3,4,9,10)) {
  cat("&", me[j], "(", std[j], ")" ) 
}

## TNR
for (j in c(5,6,11,12)) {
  cat("&", me[j], "(", std[j], ")" ) 
}
## computation
for (j in c(13,14)) {
  cat("&", me[j], "(", std[j], ")" ) 
}


# write.csv(result, "mixdata/ccares_mean.csv", row.names = FALSE, col.names = FALSE)



