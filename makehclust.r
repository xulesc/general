## makehclust.r

## load necessary libraries
library(reshape2)

## get command line arguments
args <- commandArgs(TRUE)
if (length(args) < 3) {print('no data file specified'); quit()}
data_file_name <- args[1]
distance_type <- args[2]
hclust_method <- args[3]

## load data
data.raw <- read.csv(data_file_name, header=FALSE)
data.mat <- acast(data.raw, V1~V2, value.var="V3")
data.d <- dist(data.mat, method = distance_type)

## cluster
data.fit <- hclust(data.d, method = hclust_method)

## 
print(data.fit)



