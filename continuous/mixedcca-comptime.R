time01 = rep(0,num_run)
time02 = rep(0,num_run)
Itr01 = rep(0,num_run)
Itr02 = rep(0,num_run)
tol = 0.0001
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
  time01[k] = as.numeric (Sys.time() - time1, units = "secs")
  Itr01[k] = dim(mixedCCAresult1$fitresult$w1trace)[2]
  time2 = Sys.time()
  if(Init){
    mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "continuous", w1init = w[1:p1],
                                w2init = w[(p1+1):p], tol = tol, BICtype = 2, maxiter = maxiter, trace = T)
  }else{
    mixedCCAresult2 <- mixedCCA(X, Y, type1 = "continuous", type2 = "continuous",
                                tol = tol, BICtype = 2, maxiter = maxiter, trace = T)
    
  }  
  time02[k] = as.numeric (Sys.time() - time2, units = "secs")
  Itr02[k] = dim(mixedCCAresult2$fitresult$w1trace)[2]
  #mixedCCAresult <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", BICtype = 2)

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

mean(Itr01)
mean(time01)
mean(Itr02)
mean(time02)
