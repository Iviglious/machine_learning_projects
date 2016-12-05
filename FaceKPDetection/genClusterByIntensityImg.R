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
for(i in 1){
	  print('Patches calculating...')
	  patches <- foreach (k = 1:nrow(d.train), .combine=rbind) %do% {
	    im  <- matrix(data = im.train[k,], nrow=96, ncol=96)
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
	      NA
	    }
	  }
	  patches.df <- data.frame(patches)
	  patches.nna <- patches[!is.na(patches.df[, 1]), ]
	  patches.cluster <- kmeans(patches.nna, KMEANS_K)
	  #patches.cluster <- fanny(patches.nna, KMEANS_K)
	  d.train.patches <- d.train[!is.na(patches.df[, 1]), c(i,i+1)]
	  d.train.patches[3] <- as.factor(patches.cluster$cluster)
	  colnames(d.train.patches) <- c('x', 'y', 'col')
	  
	  patches.cluster.plot <- ggplot(d.train.patches, aes(x=96-x, y=96-y, color=col)) + geom_point(shape=1)
	  ggsave(patches.cluster.plot, file= paste(out.cluster.file, (i+1)/2, "_", sub("_x", "", names(d.train)[i]), ".png", sep = ""))
	  for (j in 1:4){
	    image(1:(2 * PATCH_SIZE + 1), 1:(2 * PATCH_SIZE + 1), matrix(data = rev(colMeans(patches.nna[patches.cluster$cluster == j, ])), nrow = 2 * PATCH_SIZE + 1, ncol = 2 * PATCH_SIZE + 1), col = gray(0:255/255)) 
	 }
}
proc.time() - ptm
