library(doMC)
library(cluster)
library(ggplot2)
registerDoMC()

load('dataset/dataset.Rd')

data.dir <- 'dataset/'
out.cluster.file<- paste0(data.dir, 'outImg/cluster/')

if(!dir.exists(out.cluster.file)){
	dir.create(out.cluster.file, recursive = TRUE)
}


patch_size <- 10
op <- par(mfrow=c(3,15))
idx = seq(1, ncol(d.train), by = 2)
re <- list()
for(i in idx){
	  print('Patches calculating...')
	  patches <- foreach (k = 1:nrow(d.train), .combine=rbind) %do% {
	    im  <- matrix(data = im.train[k,], nrow=96, ncol=96)
	    x   <- d.train[k, i]
	    y   <- d.train[k, i+1]
	    x1  <- (x-patch_size)
	    x2  <- (x+patch_size)
	    y1  <- (y-patch_size)
	    y2  <- (y+patch_size)
	    if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
	    {
	      as.vector(im[x1:x2, y1:y2])
	    }
	    else
	    {
	      NA
	    }
	  }
	  d_patcher <- data.frame(patches)
	  na_v <- is.na(d_patcher[, 1])
	  clusters <- kmeans(patches[!na_v, ], 3)
	  d.train.s <- d.train[!na_v, c(i,i+1)]
	  d.train.s[3] <- as.factor(clusters$cluster)
	  colnames(d.train.s) <- c('x', 'y', 'col')
	  s <- ggplot(d.train.s, aes(x=96-x, y=96-y, color=col)) + geom_point(shape=1)
	  ggsave(s, file= paste(out.cluster.file, (i+1)/2, "_", sub("_x", "", names(d.train)[i]), ".png", sep = ""))
	  
	  
	  
	  #image(1:96, 1:96, matrix(255, 96, 96), col="white")
	  #for(j in 1:nrow(d.train.s)){
	   # if(!is.na(clusters$cluster[j]))
	   # {  
	   #   if(clusters$cluster[j] == 1){
	   #     points(96-d.train.s[j, i], 96-d.train.s[j, i+1], col = "red")
	   #   }
	   #   else if(clusters$cluster[j] == 2){
	   #     points(96-d.train.s[j, i], 96-d.train.s[j, i+1], col = "blue")
	   #   }
	   #   else if(clusters$cluster[j] == 3){
	   #     points(96-d.train.s[j, i], 96-d.train.s[j, i+1], col = "green")
	   #   }
	   # }
	  #}
	  #image(1:21, 1:21, matrix(data = rev(colMeans(patches[clusters$cluster == 1, ])), nrow = 21, ncol = 21), col = gray(0:255/255))
	  #image(1:21, 1:21, matrix(data = rev(colMeans(patches[clusters$cluster == 2, ])), nrow = 21, ncol = 21), col = gray(0:255/255))
	  #image(1:21, 1:21, matrix(data = rev(colMeans(patches[clusters$cluster == 3, ])), nrow = 21, ncol = 21), col = gray(0:255/255))
	  
}


# image(1:21, 1:21, matrix(data = rev(colMeans(patches[kkk$cluster == 3, ])), nrow = 21, ncol = 21), col = gray(0:255/255))
