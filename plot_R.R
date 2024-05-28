library(ggplot2)
library(MLmetrics)
library(data.table)
library(dplyr)

#data <- read.csv("evaluations/asia_bnl/2023-11-07 09:33:56.618510/predictions.csv", stringsAsFactors = FALSE)

## load data

allfiles <- list.files(path = "results_good/", pattern = "predictions.csv", recursive = TRUE, full.names = TRUE)

alldata <- lapply(allfiles, read.csv, stringsAsFactors = FALSE)


sachs_estimated <- read.csv("sachs_estimated/sachs_estimated/2024-05-27 15:59:10.012169/predictions.csv", stringsAsFactors = FALSE)
sachs_gt <- read.csv("results_good/gpt_35/sachs/2024-05-27 16:12:03.647301/predictions.csv", stringsAsFactors = FALSE)$answ
sachs_estimated$pred <- sachs_estimated$answ
sachs_estimated$answ <- sachs_gt
sachs_estimated$wpred <- sachs_estimated$stat_yes <- sachs_estimated$stat_no <- sachs_estimated$pred
sachs_estimated$model <- "data"
sachs_estimated$method <- "data"
sachs_estimated$data <- "sachs"
sachs_estimated$temperature


alldata <- c(alldata, list(sachs_estimated))
DDD <- reshape2::melt(alldata, measure = c("pred", "stat_yes", "stat_no", "wpred"), 
                      value.name = "prediction", variable.name = "method")


RES <- DDD %>% group_by(model, data, method, temperature) %>% summarise(accuracy = Accuracy(prediction, answ),
                                          f1 = F1_Score(prediction, answ, positive = "YES"), 
                                          precision = Precision(prediction, answ, positive = "YES"),
                                          recall = Recall(prediction, answ, positive = "YES")) 



#ggplot(RES) + 
##  geom_col(aes(x  = model, y = value)) +
#  facet_grid(rows = vars(data), cols = vars(variable), scales = "free")


### make table #### 
library(knitr)
RES %>% filter(method %in% c("pred", "stat_yes", "stat_no", "data") & is.na(temperature)) %>%
  select(data, model, method, accuracy, f1, precision) %>%
  group_by(data) %>% arrange(desc(data)) %>% 
  xtable::xtable() %>% print(booktabs = TRUE, include.rownames = FALSE,
                             floating = FALSE, file = "table_results_cis.tex")


