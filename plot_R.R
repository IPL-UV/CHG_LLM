library(ggplot2)
library(MLmetrics)
library(data.table)
library(dplyr)

#data <- read.csv("evaluations/asia_bnl/2023-11-07 09:33:56.618510/predictions.csv", stringsAsFactors = FALSE)

## load data

allfiles <- list.files(path = "results/", pattern = "predictions.csv", recursive = TRUE, full.names = TRUE)

alldata <- lapply(allfiles, read.csv, stringsAsFactors = FALSE)


DDD <- reshape2::melt(alldata, measure = c("pred", "stat_yes", "stat_no", "wpred"), 
                      value.name = "prediction", variable.name = "method")


RES <- DDD %>% group_by(model, data, method, temperature) %>% summarise(accuracy = Accuracy(prediction, answ),
                                          f1 = F1_Score(prediction, answ, positive = "YES"), 
                                          precision = Precision(prediction, answ, positive = "YES"),
                                          recall = Recall(prediction, answ, positive = "YES")) 



ggplot(RES) + 
  geom_col(aes(x  = model, y = value)) +
  facet_grid(rows = vars(data), cols = vars(variable), scales = "free")


library(knitr)
RES[, c(2,1,3,4, 5, 6, 7)] %>% filter(method %in% c("pred", "stat_yes")) %>% 
  group_by(data) %>% arrange(desc(data)) %>% kable()


