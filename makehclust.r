## makehclust.r

## load necessary libraries
library(reshape2)
library(gplots)

## get command line arguments
args <- commandArgs(TRUE)
if (length(args) < 3) {print('no data file specified'); quit()}
data_file_name <- args[1]
distance_type <- args[2]
hclust_method <- args[3]
eps_file_name <- args[4]
plot_legend <- args[5]

## load data
data.raw <- read.csv(data_file_name, header=FALSE)
data.mat <- acast(data.raw, V1~V2, value.var="V3")
data.d <- dist(data.mat, method = distance_type)

## cluster
data.fit <- hclust(data.d, method = hclust_method)

## plot to file
postscript(eps_file_name)
plot(as.dendrogram(data.fit), horiz=TRUE)
smartlegend(x = "left", y = "bottom", legend = plot_legend)
dev.off()




