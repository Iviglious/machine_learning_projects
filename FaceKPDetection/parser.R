library(doMC)
registerDoMC()

# Data input

data.dir <- 'dataset/'

train.file <- paste0(data.dir, 'training.csv')
test.file  <- paste0(data.dir, 'test.csv')
out.rdata <- paste0(data.dir, 'dataset.Rd')
d.train    <- read.csv(train.file, stringsAsFactors=F)
d.test     <- read.csv(test.file,  stringsAsFactors=F)
im.train   <- foreach(im = d.train$Image, .combine=rbind) %dopar% {
	as.integer(unlist(strsplit(im, " ")))
}
im.test    <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
	as.integer(unlist(strsplit(im, " ")))
}

d.train$Image <- NULL
d.test$Image  <- NULL

save(d.train, im.train, d.test, im.test, file= out.rdata)
