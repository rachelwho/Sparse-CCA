for(k in 1:num_run){
  i = 1
  data = read.csv( paste0("../continuous/data/", path,"data",k-1,".csv") ,header = FALSE)
  for(C in c(-2, -1, 0)){
    Rhat <- read.csv(paste0("data/", path,"R",k-1, "_",C,".csv"), header = TRUE, row.names = 1)
    X = data[,1:p1]
    Y = data[,(p1+1):(p1+p2)]
    Y = Y-C
    Y[Y<0] = 0
    if(Init){
      w = as.numeric(read.csv(paste0("data/", path,"/minit",k-1, "_",C,".csv"), header = FALSE)[,1])
    }else{
      InitF(X, Y, type1 = "continuous", type2 = "trunc",
            tol = tol, BICtype = 1, maxiter = maxiter, trace = T, KendallR = Rhat)
    }
    time1 = Sys.time()
    if(Init){
      mixedCCAresult1 <- mixedCCA(X, Y, type1 = "continuous", type2 = "trunc", w1init = w[1:p1],
                                  w2init = w[(p1+1):p], KendallR = Rhat,  tol = tol, BICtype = 1, maxiter = maxiter, trace = FALSE)
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
                                  w2init = w[(p1+1):p], KendallR = Rhat, tol = tol, BICtype = 2, maxiter = maxiter, trace = F)
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
    iter[[i]][k, 2]  = dim(mixedCCAresult$fitresult$w1trace)[2]
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
for (i in 1:3) {
  for (j in c(13,14)) {
    cat("&", me[i,j], "(", std[i,j], ")" ) 
  }
  cat('\n')
}

