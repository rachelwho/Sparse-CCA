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


# generate Rhat


for(k in 1:num_run){
  i = 1
  data = read.csv( paste0("../continuous/data/", path,"data",k-1,".csv") ,header = FALSE)
  for(C in c(-2, -1, 0)){
    X = data[,1:p1]
    Y = data[,(p1+1):(p1+p2)]
    Y = Y-C
    Y[Y<0] = 0
    time1 = Sys.time()
    e_sig2 = estimateR_mixed(X, Y,  type1 = "continuous", type2 = "trunc", use.nearPD = TRUE, nu = 0.01)
    write.csv(e_sig2[[5]],paste0("data/", path,"R_PD",k-1, "_",C,".csv"))
    time2 = Sys.time()
    print(time2 - time1)
    i = i+1
  }
}

