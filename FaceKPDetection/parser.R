
FKD_load <- function(TRAIN.FILE, TEST.FILE){
	library(doMC)
	registerDoMC()

	d.train    <- read.csv(TRAIN.FILE, stringsAsFactors=F)
	d.test    <- read.csv(TEST.FILE, stringsAsFactors=F)
	im.train   <- foreach(im = d.train$Image, .combine=rbind) %dopar% { as.integer(unlist(strsplit(im, " "))) }
	im.test    <- foreach(im = d.test$Image, .combine=rbind) %dopar% { as.integer(unlist(strsplit(im, " "))) }

	d.train$Image <- NULL
	d.test$Image <- NULL

	return(list("d.train" = d.train, "im.train" = im.train, "im.test" = im.test))
}
