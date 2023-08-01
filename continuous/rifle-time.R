

tol = 0.0001
for(k in 81:num_run){
  # Use rifle to improve the leading generalized eigenvector
  init_r <- as.numeric(read.table(paste0("data/", path, "rifleinit",k-1,".csv"))[,1])
  size = length(init_r)
  init_r <- v + rnorm(size, sd = 0.2)
  Ahat <- as.matrix(read.csv( paste0("data/", path,"Ahat",k-1,".csv") ,header = FALSE))
  Bhat <- as.matrix(read.csv( paste0("data/", path,"Bhat",k-1,".csv") ,header = FALSE))
  
  # Pick k such that the generalized eigenvector is sparse
  
  # Rifle 1
  
  final_try <- rifle(A = Ahat,B = Bhat,init_r,k = chosen,eta = 0.01,convergence = 0, maxiter = 2000)
  w = final_try$x
  w1 = (w[1:p1] / norm(w[1:p1], "2"))
  w2 = (w[(p1+1):(p1+p2)] / norm(w[(p1+1):(p1+p2)], "2"))
  errx = min(norm(vx- w1, "2")^2, norm(vx+ w1, "2")^2)
  erry = min(norm(vy- w2, "2")^2, norm(vy+ w2, "2")^2)
  start = Sys.time()
  final <- rifle(A = Ahat,B = Bhat,init_r,k = chosen,eta = 0.01,convergence = tol, maxiter = maxiter, errx, erry)
  end = Sys.time()
  time[k] = as.numeric (end - start, units = "secs")
  Itr[k] = final$iter
  w = final$x
  iter = final$iter

  
  w1 = (w[1:p1] / norm(w[1:p1], "2"))
  w2 = (w[(p1+1):(p1+p2)] / norm(w[(p1+1):(p1+p2)], "2"))
  ## squared l2 distance
  res[k, 1] = min(norm(vx- w1, "2")^2, norm(vx+ w1, "2")^2)
  res[k, 2] = min(norm(vy- w2, "2")^2, norm(vy+ w2, "2")^2)
  ## TPR & TNR
  res[k, 3] = sum(w1[vx != 0] != 0)/3
  res[k, 4] = sum(w1[vy != 0] != 0)/3
  res[k, 5] = sum(w1[vx == 0] == 0) / (p1-3)
  res[k, 6] = sum(w2[vy == 0]  == 0)/ (p2-3)
  ## expected out of sample correlation
  # res[[i]][k, 3] = w1%*%Sig12%*%w2 / sqrt(w1%*%Sig%*%w1) / sqrt(w2%*%Sig%*%w2)
  
  
  cat(k,'\n', res[k,], '\n')
  cat(Itr[k],'\n')
}

mean(Itr)
mean(time)
mean(res[,1])
mean(res[,2])
rifle <-
  function(A,B,init,k,eta=0.01,convergence=1e-3,maxiter=5000, errx = 0, erry = 0){
    W = matrix(0, nrow = maxiter+1, ncol = p)
    p <- ncol(B)
    x <- init
    x <- init/sqrt(sum(x^2))
    criteria <- 1e10
    iter <- 0
    W[iter,] = x
    while(criteria > convergence && iter <= maxiter){
      rho <- as.numeric(t(x)%*%A%*%x/(t(x)%*%B%*%x))
      C <- diag(1,p,p)+ eta/rho*(A-rho*B)
      xprime <- C%*%x/sqrt(sum((C%*%x)^2))
      # Perform truncation
      truncate.value <- sort(abs(xprime),decreasing=TRUE)[k]
      xprime[which(abs(xprime)<truncate.value)] <- 0
      xprime <- xprime/sqrt(as.numeric(t(xprime)%*%xprime))
      w <- xprime
      w1 = (w[1:p1] / norm(w[1:p1], "2"))
      w2 = (w[(p1+1):(p1+p2)] / norm(w[(p1+1):(p1+p2)], "2"))
      err0 = min(norm(vx- w1, "2")^2, norm(vx+ w1, "2")^2)
      err1 = min(norm(vy- w2, "2")^2, norm(vy+ w2, "2")^2)
      criteria <- max(err0 - errx, err1 - erry)
      x <- xprime
      iter <- iter+1
      W[iter,] = x
    }
    
    return(list(x=xprime, W = W[1:iter,], iter = iter))
  }
