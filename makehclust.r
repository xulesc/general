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

doHClust <- function(f) {
	data.raw <- read.csv(f,header=FALSE)
	data.d <- as.dist(acast(data.raw, V1~V2, value.var="V3"))
	data.fit <- hclust(data.d, method=hclust_method)
	return(data.fit)
}

writeClusterMembership <- function(fitHclust, fname, kcount) {
	clusterId <- cutree(fitHclust, k=kcount)
	write.table(data.frame(clusterId), file=fname, col.names=F, 
		row.names=T, sep=",")	
}

## cluster (assume data file contains pairwise distances)
data.fit <- doHClust(data_file_name)

## write clusters
writeClusterMembership(data.fit, 'data.clust.csv',5)

## plot to file
postscript(eps_file_name)
plot(as.dendrogram(data.fit), horiz=TRUE)
smartlegend(x = "left", y = "bottom", legend = plot_legend)
dev.off()




