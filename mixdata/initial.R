library(dplyr)
library(mixedCCA)
setwd("/projectnb/labci/Qiuyun/SCCA/v2/mixdata")
# setwd("/Users/apple/Desktop/Dropbox/Sparse CCA/simulation/v2/continuous")

path = "100-100_180_0.7_0.8/"
tol = 0.00001
maxiter = 5000
record_trace = FALSE
Init = FALSE
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


### Default initial starting point
InitF <- function(X1, X2, type1, type2,
                  lamseq1 = NULL, lamseq2 = NULL, nlamseq = 20, lam.eps = 1e-02,
                  w1init = NULL, w2init = NULL, BICtype,
                  KendallR,
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
  
  write.table(c(w1init, w2init), sep = ',', file = paste0("data/", path,"/minit_exact_SPD",k-1,"_",C,".csv"), col.names = FALSE, row.names = FALSE)
}


for(k in 1:num_run){
  i = 1
  data = read.csv( paste0("../continuous/data/", path,"data",k-1,".csv") ,header = FALSE)
  for(C in c(-2, -1, 0)){
    Rhat_PD <- read.csv(paste0("data/", path,"R_PD",k-1, "_",C,".csv"), header = TRUE, row.names = 1)
    X = data[,1:p1]
    Y = data[,(p1+1):(p1+p2)]
    Y = Y-C
    Y[Y<0] = 0
    InitF(X, Y, type1 = "continuous", type2 = "trunc",
          tol = tol, BICtype = 1, maxiter = maxiter, trace = T, KendallR = Rhat_PD)
  }
}


