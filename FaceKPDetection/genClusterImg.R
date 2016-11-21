load('dataset/dataset.Rd')

data.dir <- 'dataset/'
out.cluster.file<- paste0(data.dir, 'outImg/cluster/')

if(!dir.exists(out.cluster.file)){
	dir.create(out.cluster.file, recursive = TRUE)
}

idx = seq(1, ncol(d.train), by = 2)
for(i in idx){
	png(paste(out.cluster.file, (i+1)/2, "_", sub("_x", "", names(d.train)[i]), ".png", sep = ""))
	image(1:96, 1:96, matrix(255, 96, 96), col="white")
	title(paste((i+1)/2, "_", sub("_x", "", names(d.train)[i]), sep = ""))
	for(j in 1:nrow(d.train)){
		points(96-d.train[j, i], 96-d.train[j, i+1], col = "red")
	}
	dev.off()
}
