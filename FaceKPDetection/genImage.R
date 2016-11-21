library(doMC)
registerDoMC()

# Data input

load('dataset/dataset.Rd')
data.dir <- 'dataset/'
out.train.file  <- paste0(data.dir, 'outImg/train/')
out.test.file<- paste0(data.dir, 'outImg/test/')

if(!dir.exists(out.train.file)){
	dir.create(out.train.file, recursive = TRUE)
}
if(!dir.exists(out.test.file)){
	dir.create(out.test.file, recursive = TRUE)
}
idx = seq(1, ncol(d.train), by = 2)
for (i in 1:nrow(im.train)){
	print(paste("Image: " , i, " ...done"))
	png(filename= paste(out.train.file, i, ".png", sep = ""))
	image(1:96, 1:96, matrix(data=rev(im.train[i,]), nrow=96, ncol=96), col=gray((0:255)/255))
	for (j in idx){
		if (!is.na(d.train[i, j]) && !is.na(d.train[i, j+1])){
			points(96-d.train[i, j], 96-d.train[i, j+1], col = "red")
		}
	}
	dev.off()
}
for (i in 1:nrow(im.test)){
	print(paste("Image: " , i, " ...done"))
	png(filename= paste(out.test.file, i, ".png", sep = ""))
	image(1:96, 1:96, matrix(data=rev(im.test[i,]), nrow=96, ncol=96), col=gray((0:255)/255))
	dev.off()
}
