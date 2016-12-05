rm(list=ls())
# Library -----------------------------------------------------------------


library(doMC)
library(cluster)
library(ggplot2)
registerDoMC()


# Parameter ---------------------------------------------------------------


data.dir <- 'dataset/'
out.cluster.file<- paste0(data.dir, 'outImg/cluster/')
PATCH_SIZE <- 10
KMEANS_K <- 6

# Load/Save data ---------------------------------------------------------------


load('dataset/dataset.Rd')
if(!dir.exists(out.cluster.file)){
  dir.create(out.cluster.file, recursive = TRUE)
}

# Generate cluster image --------------------------------------------------


idx = seq(1, ncol(d.train), by = 2)
ptm <- proc.time()
for(i in idx){
  d.train.sub <- d.train[, c(i,i+1)]
  d.train.sub.nna <- d.train.sub[!is.na(d.train.sub[, 1]), ]
  im.train.sub.nna <- im.train[!is.na(d.train.sub[, 1]), ]
	  d.train.cluster <- kmeans(d.train.sub.nna, KMEANS_K)
	  #patches.cluster <- fanny(patches.nna, KMEANS_K)
	  d.train.cluster.data <- d.train.sub.nna
	  d.train.cluster.data[3] <- as.factor(d.train.cluster$cluster)
	  colnames(d.train.cluster.data) <- c('x', 'y', 'col')

	  d.train.cluster.plot <- ggplot(d.train.cluster.data, aes(x=96-x, y=96-y, color=col)) + geom_point(shape=1)
	  ggsave(d.train.cluster.plot, file= paste(out.cluster.file, (i+1)/2, "_", sub("_x", "", names(d.train)[i]), ".png", sep = ""))
	  for (j in 1:KMEANS_K){
	    patches <- foreach (k = 1:nrow(d.train.sub.nna), .combine=rbind) %do% {
	      if (d.train.cluster.data[k, 3] == j){
	        im  <- matrix(data = im.train.sub.nna[k,], nrow=96, ncol=96)
	        x   <- d.train[k, i]
	        y   <- d.train[k, i+1]
	        x1  <- (x-PATCH_SIZE)
	        x2  <- (x+PATCH_SIZE)
	        y1  <- (y-PATCH_SIZE)
	        y2  <- (y+PATCH_SIZE)
	        if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) ){
	          as.vector(im[x1:x2, y1:y2])
	        }
	        else{
	          NULL
	        }
	      }
	      else{
	        NULL
	      }
	    }
	    image(1:(2 * PATCH_SIZE + 1), 1:(2 * PATCH_SIZE + 1), matrix(data = rev(colMeans(patches)), nrow = 2 * PATCH_SIZE + 1, ncol = 2 * PATCH_SIZE + 1), col = gray(0:255/255))
	 }
}
proc.time() - ptm
