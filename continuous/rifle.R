library(rifle)
## Example on CCA
setwd(here::here())
path = "250-250_0.7_0.8/"
tol = 0.001
chosen <- 12
record_trace = TRUE
maxiter = 1000
dir.create(file.path(getwd(), paste0("result/", path, 'rifle')), showWarnings = FALSE)

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
ind = 6

v = c(vx, vy)

time = rep(0, num_run)
Itr = rep(0, num_run)
res = matrix(0, nrow = num_run, ncol = ind)

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

## Run initialization stage
for(k in 1:num_run){
  data = read.csv( paste0("data/", path,"data",k-1,".csv") ,header = FALSE)
  n = length(data)
  Ahat <- as.matrix(read.csv( paste0("data/", path,"Ahat",k-1,".csv") ,header = FALSE))
  Bhat <- as.matrix(read.csv( paste0("data/", path,"Bhat",k-1,".csv") ,header = FALSE))
  start = Sys.time()
  # Running initialization using convex relaxation
  a <- initial.convex(A=Ahat,B=Bhat,lambda=2*sqrt(log(p)/n),K=1,nu=1,trace=TRUE, maxiter = Inf)
  init_r <- eigen(a$Pi+t(a$Pi))$vectors[,1]
  print(cal_error(init_r, v/ norm(v, "2")))
  #  ## computation time
  time[k] = as.numeric (Sys.time() - start, units = "secs")
  # write.table(init_r,paste0("data/", path, "rifleinit",k-1,".csv"), row.names = FALSE, col.names = FALSE)
  # Itr[k] = a$iteration
  # print(time[k])
  # write.table(time,paste0("result/", path, "rifle/init_time.csv"), row.names = FALSE, col.names = FALSE)
  # write.table(Itr,paste0("result/", path, "rifle/init_itr.csv"), row.names = FALSE, col.names = FALSE)
}

rifle <-
  function(A,B,init,k,eta=0.01,convergence=1e-3,maxiter=5000){
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
      criteria <- sqrt(sum((x-xprime)^2))
      x <- xprime
      iter <- iter+1
      W[iter,] = x
      print(iter)
      print(criteria)
    }
    
    return(list(x=xprime, W = W[1:iter,], iter = iter))
  }

for(k in 1:num_run){
  # Use rifle to improve the leading generalized eigenvector
  init_r <- as.numeric(read.table(paste0("data/", path, "rifleinit",k-1,".csv"))[,1])
  w = init_r
  w1 = (w[1:p1] / norm(w[1:p1], "2"))
  w2 = (w[(p1+1):(p1+p2)] / norm(w[(p1+1):(p1+p2)], "2"))
  ## squared l2 distance
  res[k, 1] = min(norm(vx- w1, "2")^2, norm(vx+ w1, "2")^2)
  res[k, 2] = min(norm(vy- w2, "2")^2, norm(vy+ w2, "2")^2)
}
mean(res[,1], na.rm = TRUE)
mean(res[,2], na.rm = TRUE)
sum(is.na(res[,2]))
### run rifle
for(k in 1:num_run){
  # Use rifle to improve the leading generalized eigenvector
  # init_r <- as.numeric(read.table(paste0("data/", path, "rifleinit",k-1,".csv"))[,1])
  # size = length(init_r)
  init_r <- v + rnorm(size, sd = 0.2)
  Ahat <- as.matrix(read.csv( paste0("data/", path,"Ahat",k-1,".csv") ,header = FALSE))
  Bhat <- as.matrix(read.csv( paste0("data/", path,"Bhat",k-1,".csv") ,header = FALSE))
  
  # Pick k such that the generalized eigenvector is sparse
  
  # Rifle 1
  start = Sys.time()
  final <- rifle(A = Ahat,B = Bhat,init_r,k = chosen,eta = 0.01,convergence = tol, maxiter = maxiter)
  end = Sys.time()
  
  w = final$x
  iter = final$iter
  if(record_trace == TRUE){
    error = matrix(nrow = 7, ncol = iter)
    W = final$W
    error[1,] = apply(W[,1:p1], 1, cal_error,v = vx)
    error[2,] = apply(W[,(p1+1):(p1+p2)], 1, cal_error, v = vy)
    error[3,] = apply(W[,1:p1], 1, cal_tpr,v = vx)
    error[4,] = apply(W[,(p1+1):(p1+p2)], 1, cal_tpr, v = vy)
    error[5,] = apply(W[,1:p1], 1, cal_tnr,v = vx)
    error[6,] = apply(W[,(p1+1):(p1+p2)], 1, cal_tnr, v = vy)
    error[7,] = seq(0,as.numeric (end - start, units = "secs"), length.out = iter)
    write.table(error,paste0("result/", path, "rifle/error-time",k-1,".csv"), row.names = FALSE, col.names = FALSE)
  }

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
}
write.table(res,paste0("result/", path, "rifle/error.csv"), row.names = FALSE, col.names = FALSE)

rr = read.table(paste0("result/", path, "rifle/error.csv"))
rr = res
## squared l2 distance
me= round(apply(rr, 2, mean),2)
std = round(apply(rr, 2, sd) ,2)
for (j in c(1,2)) {
  cat("&", me[j], "(", std[j], ")" ) 
}
## TPR
for (j in c(3,4)) {
  cat("&", me[j], "(", std[j], ")" ) 
}

## TNR
for (j in c(5,6)) {
  cat("&", me[j], "(", std[j], ")" ) 
}



# initial.convex <-
#   function(A,B,lambda,K,nu=1,epsilon=5e-3,maxiter=1000,trace=FALSE){
#     p <- nrow(B)
#     eigenB <- eigen(B)
#     sqB <- eigenB$vectors%*%sqrt(diag(pmax(eigenB$values,0)))%*%t(eigenB$vectors)	
#     tau <- 4*nu*eigenB$values[1]^2	
#     criteria <- 1e10
#     i <- 1
#     # Initialize parameters
#     H <- Pi <- oldPi <-  diag(1,p,p)
#     Gamma <- matrix(0,p,p)
#     # While loop for the iterations
#     while(criteria > epsilon && i <= maxiter){
#       Pi <- updatePi(B,sqB,A,H,Gamma,nu,lambda,Pi,tau)
#       
#       H <- updateH(sqB,Gamma,nu,Pi,K)
#       Gamma <- Gamma + sqB%*%Pi%*%sqB-H	
#       criteria <- sqrt(sum((Pi-oldPi)^2))
#       oldPi <- Pi
#       i <- i+1
#       if(trace==TRUE)
#       {
#         print(i)
#         print(criteria)
#       }
#     }
#     return(list(Pi=Pi,H=H,Gamma=Gamma,iteration=i,convergence=criteria))
#     
#   }
