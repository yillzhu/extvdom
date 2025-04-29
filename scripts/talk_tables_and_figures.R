# Clear workspace

rm(list = ls())

# Define exit function

exit <- function() { invokeRestart("abort") } 

# Load libraries

library(readr)
library(janitor)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(scales)
library(xtable)
library(ggridges)
library(gridExtra)
library(forcats)

# Read data and clean up

if(!file.exists("../results/output.csv")) {
    list_of_files <- list.files(path = "../results/hpc_csv", recursive = TRUE,
                                pattern = "\\.csv$", full.names = TRUE)
    tic <- 1
    ind <- 1
    df <- c()
    while(tic <= length(list_of_files)) {
        toc <- min(tic + 500, length(list_of_files))
        df[[ind]] <- read_csv(list_of_files[tic : toc])
        tic <- toc + 1
        ind <- ind + 1
    }
    df <- do.call(rbind, df)
    write_csv(df, "../results/output.csv")
} else {
    df <- read_csv("../results/output.csv")
}

df$v_domain <- factor(df$v_domain, c("bestsample", "box", "ch", "isofor", "chplus", "svm", "svmplus",
    "svm0", "svm0plus", "ae", "aeplus"))

# Sort

df <- df %>% arrange(seed, func, noise, v_domain)

# Filter as necessary

# df <- filter(df, sampling == "normal_at_min")
# df <- filter(df, noise == 0.0)
df <- filter(df, v_domain != "svm" & v_domain != "svmplus")
df <- filter(df, v_domain != "bestsample")

# Add global optimal values

df$gval <- NA
df$gval[df$func == "beale"] <- 0
df$gval[df$func == "peaks"] <- -6.55113
df$gval[df$func == "griewank"] <- 0
df$gval[df$func == "powell"] <- 0
df$gval[df$func == "quintic"] <- 0
df$gval[df$func == "qing"] <- 0
df$gval[df$func == "rastrigin"] <- 0
df$gval[df$func == "rastrigin2d"] <- 0
df$gval[df$func == "rastrigin10d"] <- 0
df <- df %>% rename(truval_trusol = gval)

# Add optval_err

df$optval_err <- abs(df$modval_modsol - df$truval_trusol)

results_mean <- df %>%
    group_by(func, v_domain, sampling) %>%
    summarize(count = n(),
              mean_funval_err = mean(funval_err),
              mean_optval_err = mean(abs(modval_modsol - truval_trusol)),
              mean_optsol_err = mean(optsol_err)
    ) %>%
    arrange(func, sampling, v_domain)

results_sd <- df %>%
    group_by(func, v_domain, sampling) %>%
    summarize(count = n(),
        sd_funval_err = sd(funval_err),
        sd_optval_err = sd(abs(modval_modsol - truval_trusol)),
        sd_optsol_err = sd(optsol_err)
        ) %>%
    arrange(func, sampling, v_domain)

##

med_funval_err <- df %>%
    filter(func == "beale", sampling == "uniform") %>%
    group_by(v_domain, learning) %>%
    summarize(med = median(funval_err)) %>%
    pivot_wider(names_from = learning, values_from = med)

################################################################################

# Draw some pics

df <- read_csv("../results/output.csv")
df$v_domain <- factor(df$v_domain, c("bestsample", "box", "ch", "isofor", "chplus", "svm", "svmplus",
    "svm0", "svm0plus", "ae", "aeplus"))
df <- df %>% arrange(seed, func, noise, v_domain)
df$gval <- NA
df$gval[df$func == "beale"] <- 0
df$gval[df$func == "peaks"] <- -6.55113
df$gval[df$func == "griewank"] <- 0
df$gval[df$func == "powell"] <- 0
df$gval[df$func == "quintic"] <- 0
df$gval[df$func == "qing"] <- 0
df$gval[df$func == "rastrigin"] <- 0
df$gval[df$func == "rastrigin2d"] <- 0
df$gval[df$func == "rastrigin10d"] <- 0
df <- df %>% rename(truval_trusol = gval)
df$optval_err <- abs(df$modval_modsol - df$truval_trusol)

df <- filter(df, v_domain != "svm" & v_domain != "svmplus")
df <- filter(df, v_domain != "bestsample")
# df <- filter(df, noise > 0)

my_plot_function <- function(my_func, my_sampling, my_learning) {
    
    tmp <- df %>%
        filter(func == my_func, sampling == my_sampling, learning == my_learning) %>%
        select(
            func, noise, sampling, sample_sz, seed, learning, v_domain,
            funval_err, optval_err, optsol_err
        ) %>%
        group_by(v_domain) %>%
        summarize(med_funval_err = median(funval_err),
                  med_optval_err = median(optval_err),
                  med_optsol_err = median(optsol_err))
    
    val <- tmp$med_funval_err[1] # box
    df$funval_err <- df$funval_err / val
    tmp[[2]] <- formatC(tmp[[2]] / val, format = "f", digits = 2)
    
    val <- tmp$med_optval_err[1] # box
    df$optval_err <- df$optval_err / val
    tmp[[3]] <- formatC(tmp[[3]] / val, format = "f", digits = 2)

    val <- tmp$med_optsol_err[1] # box
    df$optsol_err <- df$optsol_err / val
    tmp[[4]] <- formatC(tmp[[4]] / val, format = "f", digits = 2)

    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "box"] <- "Box"
    tmp$v_domain[tmp$v_domain == "ch"] <- "CH"
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    tmp$v_domain[tmp$v_domain == "isofor"] <- "IsoFor"
    names(tmp) <- c("Validity\nDomain", "Median Function\nValue Error", "Median Optimal\nValue Error", "Median Optimal\nSolution Error")
    
    fname <- paste0("../results/plots/", my_func, "_", my_sampling, "_", my_learning, "_table.png")
    png(fname, res = 300, height = 200*nrow(tmp), width = 400*ncol(tmp))
    tmp <- apply(tmp, 1:2, formatC, format="f", digits=3)
    grid.table(tmp)
    dev.off()
   
    print(my_func)
    print(my_sampling)
    print(my_learning)
    df <- filter(df, func == my_func, sampling == my_sampling, learning == my_learning)
    df$v_domain <- as.character(df$v_domain)
    df$v_domain[df$v_domain == "box"] <- "Box"
    df$v_domain[df$v_domain == "ch"] <- "CH"
    df$v_domain[df$v_domain == "chplus"] <- "CH+"
    df$v_domain[df$v_domain == "isofor"] <- "IsoFor"
    df$v_domain <- factor(df$v_domain, c("Box", "CH", "IsoFor", "CH+"))
    plot1 <- ggplot(df, aes(x = funval_err, y = fct_rev(v_domain), fill = v_domain)) +
        stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
        scale_x_continuous(trans = "log10", n.breaks = 4, labels = number) +
        theme(legend.position = "none") +
        ggtitle("Function Value Error") +
        theme(axis.title = element_blank()) +
        theme(axis.ticks = element_blank()) +
        theme(axis.text.y = element_blank())
    plot2 <- ggplot(df, aes(x = optval_err, y = fct_rev(v_domain), fill = v_domain)) +
        stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
        scale_x_continuous(trans = "log10", n.breaks = 4, labels = number) +
        theme(legend.position = "none") +
        ggtitle("Optimal Value Error") +
        theme(axis.title = element_blank()) +
        theme(axis.ticks = element_blank()) +
        theme(axis.text.y = element_blank())
    plot3 <- ggplot(df, aes(x = optsol_err, y = fct_rev(v_domain), fill = v_domain)) +
        stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
        scale_x_continuous(trans = "log10", n.breaks = 4, labels = number) +
        ggtitle("Optimal Solution Error") +
        theme(axis.title = element_blank()) +
        theme(axis.ticks = element_blank()) +
        theme(axis.text.y = element_blank()) +
        scale_fill_hue(name = "Validity\nDomain")
    
    p <- grid.arrange(plot1, plot2, plot3, nrow = 1, widths = c(1, 1, 1.3))
    fname <- paste0("../results/plots/", my_func, "_", my_sampling, "_", my_learning, ".png")
    ggsave(fname, p, width = 9, height = 4, dpi = 300)

}

for(my_func in unique(as.character(df$func))) {
    for(my_sampling in unique(as.character(df$sampling))) {
        for(my_learning in unique(as.character(df$learning))) {
            my_plot_function(my_func, my_sampling, my_learning)
        }
    }
}

exit()

# # Widen data frame so that each row corresponds to an instance (not an an instance-method pair)
#
# df_wide <- df %>%
#     select(
#         seed, func, sample_sz, noise_std, sampling, learning, v_domain,
#         predval, trueval, valerr, norm_valerr
#     ) %>%
#     pivot_wider(names_from = v_domain, values_from = c("predval", "trueval",
#         "valerr", "norm_valerr"))
#
# # Remove NAs
#
# df_wide <- df_wide[complete.cases(df_wide), ]
#
# # Add global optimal values
#
# df_wide$gval <- NA
# df_wide$gval[df_wide$func == "beale"] <- 0
# df_wide$gval[df_wide$func == "peaks"] <- -6.55113
# df_wide$gval[df_wide$func == "griewank"] <- 0
# df_wide$gval[df_wide$func == "powell"] <- 0
# df_wide$gval[df_wide$func == "quintic"] <- 0
# df_wide$gval[df_wide$func == "qing"] <- 0
# df_wide$gval[df_wide$func == "rastrigin"] <- 0
# df_wide$gval[df_wide$func == "rastrigin2d"] <- 0
#
# examine_metric <- function(metric) {
#
#     if(metric == "valerr") {
#         mydf <- df_wide %>%
#             mutate(ch_none = (1 - valerr_ch / valerr_none)) %>%
#             mutate(ch_plus_none = (1 - valerr_ch_plus / valerr_none)) %>%
#             mutate(ch_plus_ch = (1 - valerr_ch_plus / valerr_ch)) %>%
#             mutate(svm_none = (1 - valerr_svm / valerr_none)) %>%
#             mutate(svm_plus_none = (1 - valerr_svm_plus / valerr_none)) %>%
#             mutate(svm_plus_svm = (1 - valerr_svm_plus / valerr_svm)) %>%
#             select(-predval_none, -predval_ch, -predval_ch_plus, -predval_svm, -predval_svm_plus) %>%
#             select(-trueval_none, -trueval_ch, -trueval_ch_plus, -trueval_svm, -trueval_svm_plus) %>%
#             select(-norm_valerr_none, -norm_valerr_ch, -norm_valerr_ch_plus)
#     }
#
#     if(metric == "trueval") {
#         mydf <- df_wide %>%
#             mutate(ch_none = ((trueval_none - trueval_ch) / (trueval_none - gval))) %>%
#             mutate(ch_plus_none = ((trueval_none - trueval_ch_plus) / (trueval_none - gval))) %>%
#             mutate(ch_plus_ch = ((trueval_ch - trueval_ch_plus) / (trueval_ch - gval))) %>%
#             mutate(svm_none = ((trueval_none - trueval_svm) / (trueval_none - gval))) %>%
#             mutate(svm_plus_none = ((trueval_none - trueval_svm_plus) / (trueval_none - gval))) %>%
#             mutate(svm_plus_svm = ((trueval_svm - trueval_svm_plus) / (trueval_svm - gval))) %>%
#             select(-predval_none, -predval_ch, -predval_ch_plus) %>%
#             select(-valerr_none, -valerr_ch, -valerr_ch_plus) %>%
#             select(-norm_valerr_none, -norm_valerr_ch, -norm_valerr_ch_plus)
#     }
#
#     mydf$change_ch_none <- NA
#     mydf$change_ch_plus_none <- NA
#     mydf$change_ch_plus_ch <- NA
#
#     mydf$change_svm_none <- NA
#     mydf$change_svm_plus_none <- NA
#     mydf$change_svm_plus_svm <- NA
#
#     tol <- 0.01
#
#     # ch compared to none
#
#     cha <- NA
#     met <- mydf$ch_none
#     cha[abs(met) < tol] <- "same"
#     cha[met <= -tol] <- "worse"
#     cha[met >=  tol] <- "better"
#     mydf$change_ch_none <- cha
#
#     ind <- (cha == "worse")
#
#     if(metric == "valerr") {
#         mydf$ch_none[ind] <-
#             -(1 - mydf$valerr_none[ind] / mydf$valerr_ch[ind])
#     }
#
#     if(metric == "trueval") {
#         mydf$ch_none[ind] <-
#             - (mydf$trueval_ch[ind] - mydf$trueval_none[ind]) /
#             (mydf$trueval_ch[ind] - mydf$gval[ind])
#     }
#
#     # svm compared to none
#
#     cha <- NA
#     met <- mydf$svm_none
#     cha[abs(met) < tol] <- "same"
#     cha[met <= -tol] <- "worse"
#     cha[met >=  tol] <- "better"
#     mydf$change_svm_none <- cha
#
#     ind <- (cha == "worse")
#
#     if(metric == "valerr") {
#         mydf$svm_none[ind] <-
#             -(1 - mydf$valerr_none[ind] / mydf$valerr_svm[ind])
#     }
#
#     if(metric == "trueval") {
#         mydf$svm_none[ind] <-
#             - (mydf$trueval_svm[ind] - mydf$trueval_none[ind]) /
#             (mydf$trueval_svm[ind] - mydf$gval[ind])
#     }
#
#     # ch_plus compared to none
#
#     cha <- NA
#     met <- mydf$ch_plus_none
#     cha[abs(met) < tol] <- "same"
#     cha[met <= -tol] <- "worse"
#     cha[met >=  tol] <- "better"
#     mydf$change_ch_plus_none <- cha
#
#     ind <- (cha == "worse")
#
#     if(metric == "valerr") {
#         mydf$ch_plus_none[ind] <-
#             -(1 - mydf$valerr_none[ind] / mydf$valerr_ch_plus[ind])
#     }
#
#     if(metric == "trueval") {
#         mydf$ch_plus_none[ind] <-
#             - (mydf$trueval_ch_plus[ind] - mydf$trueval_none[ind]) /
#             (mydf$trueval_ch_plus[ind] - mydf$gval[ind])
#     }
#
#     # svm_plus compared to none
#
#     cha <- NA
#     met <- mydf$svm_plus_none
#     cha[abs(met) < tol] <- "same"
#     cha[met <= -tol] <- "worse"
#     cha[met >=  tol] <- "better"
#     mydf$change_svm_plus_none <- cha
#
#     ind <- (cha == "worse")
#
#     if(metric == "valerr") {
#         mydf$svm_plus_none[ind] <-
#             -(1 - mydf$valerr_none[ind] / mydf$valerr_svm_plus[ind])
#     }
#
#     if(metric == "trueval") {
#         mydf$svm_plus_none[ind] <-
#             - (mydf$trueval_svm_plus[ind] - mydf$trueval_none[ind]) /
#             (mydf$trueval_svm_plus[ind] - mydf$gval[ind])
#     }
#
#     # ch_plus compared to ch
#
#     cha <- NA
#     met <- mydf$ch_plus_ch
#     cha[abs(met) < tol] <- "same"
#     cha[met <= -tol] <- "worse"
#     cha[met >=  tol] <- "better"
#     mydf$change_ch_plus_ch <- cha
#
#     ind <- (cha == "worse")
#
#     if(metric == "valerr") {
#         mydf$ch_plus_ch[ind] <-
#             -(1 - mydf$valerr_ch[ind] / mydf$valerr_ch_plus[ind])
#     }
#
#     if(metric == "trueval") {
#         mydf$ch_plus_ch[ind] <-
#             - (mydf$trueval_ch_plus[ind] - mydf$trueval_ch[ind]) /
#             (mydf$trueval_ch_plus[ind] - mydf$gval[ind])
#     }
#
#     # svm_plus compared to svm
#
#     cha <- NA
#     met <- mydf$svm_plus_svm
#     cha[abs(met) < tol] <- "same"
#     cha[met <= -tol] <- "worse"
#     cha[met >=  tol] <- "better"
#     mydf$change_svm_plus_svm <- cha
#
#     ind <- (cha == "worse")
#
#     if(metric == "valerr") {
#         mydf$svm_plus_svm[ind] <-
#             -(1 - mydf$valerr_svm[ind] / mydf$valerr_svm_plus[ind])
#     }
#
#     if(metric == "trueval") {
#         mydf$svm_plus_svm[ind] <-
#             - (mydf$trueval_svm_plus[ind] - mydf$trueval_svm[ind]) /
#             (mydf$trueval_svm_plus[ind] - mydf$gval[ind])
#     }
#
#     mydf
#
# }
#
# # Examine percent change in valerr
#
# examine_changes <- function(mydf, v) {
#
#     v_none <- rlang::sym(paste0(v, "_none"))
#     change_v_none <- rlang::sym(paste0("change_", v, "_none"))
#
#     v_plus_none <- rlang::sym(paste0(v, "_plus_none"))
#     change_v_plus_none <- rlang::sym(paste0("change_", v, "_plus_none"))
#
#     v_plus_v <- rlang::sym(paste0(v, "_plus_", v))
#     change_v_plus_v <- rlang::sym(paste0("change_", v, "_plus_", v))
#
#     tmp1 <- mydf %>%
#         group_by(!!change_v_none) %>%
#         summarize(
#             cnt_v_none = n(),
#             avg_v_none = percent(mean(!!v_none)),
#             .groups = "drop"
#         )
#
#     tmp3 <- mydf %>%
#         group_by(!!change_v_plus_none) %>%
#         summarize(
#             cnt_v_plus_none = n(),
#             avg_v_plus_none = percent(mean(!!v_plus_none)),
#             .groups = "drop"
#         )
#
#     # tmp1 <- cbind(tmp1, tmp3[, c(2, 3)])
#     tmp1 <- merge(tmp1, tmp3, by = 1, all = TRUE)
#
#     tmp4 <- mydf %>%
#         group_by(!!change_v_plus_v) %>%
#         summarize(
#             cnt_v_plus_v = n(),
#             avg_v_plus_v = percent(mean(!!v_plus_v)),
#             .groups = "drop"
#         )
#
#     tmp1 <- cbind(tmp1, tmp4[, c(2, 3)])
#
#     names(tmp1)[1] <- "status"
#     tmp1 <- tmp1[, c(1, 2, 4, 6, 3, 5, 7)]
#
# }
#
# df_valerr <- examine_metric("valerr")
# df_trueval <- examine_metric("trueval")
#
# cat("\nch\tvalerr\n\n")
# print(examine_changes(df_valerr, "ch"))
#
# cat("\nch\ttrueval\n\n")
# print(examine_changes(df_trueval, "ch"))
#
# cat("\nsvm\tvalerr\n\n")
# print(examine_changes(df_valerr, "svm"))
#
# cat("\nsvm\ttrueval\n\n")
# print(examine_changes(df_trueval, "svm"))
#
# tmp <- df_valerr %>%
#     group_by(change_ch_none, change_ch_plus_ch) %>%
#     summarize(count = n(), perc_ch_none = percent(median(ch_none)), perc_ch_plus_ch = percent(median(ch_plus_ch)), .groups = "drop") %>%
#     arrange(desc(count))
# # print(tmp)
#
# exit()
#
#
# # Examine percent change trueval
#
# # df_trueval <- df_wide %>%
# #     mutate(ch_none = ((trueval_none - trueval_ch) / (trueval_none - gval))) %>%
# #     mutate(ch_plus_none = ((trueval_none - trueval_ch_plus) / (trueval_none - gval))) %>%
# #     mutate(ch_plus_ch = ((trueval_ch - trueval_ch_plus) / (trueval_ch - gval))) %>%
# #     select(-predval_none, -predval_ch, -predval_ch_plus) %>%
# #     select(-valerr_none, -valerr_ch, -valerr_ch_plus) %>%
# #     select(-norm_valerr_none, -norm_valerr_ch, -norm_valerr_ch_plus)
# #
# # df_trueval$change_ch_none <- NA
# # df_trueval$change_ch_plus_none <- NA
# # df_trueval$change_ch_plus_ch <- NA
# #
# # tol <- 0.01
# #
# # cha <- NA
# # met <- df_trueval$ch_none
# # cha[abs(met) < tol] <- "same"
# # cha[met <= -tol] <- "worse"
# # cha[met >=  tol] <- "better"
# # df_trueval$change_ch_none <- cha
# #
# # ind <- (cha == "worse")
# # df_trueval$ch_none[ind] <-
# #     - (df_trueval$trueval_ch[ind] - df_trueval$trueval_none[ind]) /
# #     (df_trueval$trueval_ch[ind] - df_trueval$gval[ind])
# #
# # cha <- NA
# # met <- df_trueval$ch_plus_none
# # cha[abs(met) < tol] <- "same"
# # cha[met <= -tol] <- "worse"
# # cha[met >=  tol] <- "better"
# # df_trueval$change_ch_plus_none <- cha
# #
# # ind <- (cha == "worse")
# # df_trueval$ch_plus_none[ind] <-
# #     - (df_trueval$trueval_ch_plus[ind] - df_trueval$trueval_none[ind]) /
# #     (df_trueval$trueval_ch_plus[ind] - df_trueval$gval[ind])
# #
# # cha <- NA
# # met <- df_trueval$ch_plus_ch
# # cha[abs(met) < tol] <- "same"
# # cha[met <= -tol] <- "worse"
# # cha[met >=  tol] <- "better"
# # df_trueval$change_ch_plus_ch <- cha
# #
# # ind <- (cha == "worse")
# # df_trueval$ch_plus_ch[ind] <-
# #     - (df_trueval$trueval_ch_plus[ind] - df_trueval$trueval_ch[ind]) /
# #     (df_trueval$trueval_ch_plus[ind] - df_trueval$gval[ind])
# #
# # df_trueval %>%
# #     group_by(change_ch_none) %>%
# #     summarize(
# #         count = n(),
# #         mean_change = percent(mean(ch_none)),
# #         median_change = percent(median(ch_none)),
# #         #        mean_valerr_none = mean(valerr_none),
# #         #        mean_valerr_ch = mean(valerr_ch)
# #     )
# #
# # df_trueval %>%
# #     group_by(change_ch_plus_none) %>%
# #     summarize(
# #         count = n(),
# #         mean_change = percent(mean(ch_plus_none)),
# #         median_change = percent(median(ch_plus_none)),
# #         #        mean_valerr_none = mean(valerr_none),
# #         #        mean_valerr_ch_plus = mean(valerr_ch_plus)
# #     )
# #
# # df_trueval %>%
# #     group_by(change_ch_plus_ch) %>%
# #     summarize(
# #         count = n(),
# #         mean_change = percent(mean(ch_plus_ch)),
# #         median_change = percent(median(ch_plus_ch)),
# #         #        mean_valerr_none = mean(valerr_none),
# #         #        mean_valerr_ch_plus = mean(valerr_ch_plus)
# #     )
#
#
# ################################################################################
# # End work on 2023-10-23
# ################################################################################
#
# # Examine change in trueval
#
# df_trueval <- df_wide %>%
#     select(seed, func, trueval_none, trueval_ch, trueval_ch_plus)
#
# tol <- 1.0e-4
#
# df_trueval <- df_trueval %>%
#     mutate(ch_better_none = as.integer(trueval_ch + tol < trueval_none)) %>%
#     mutate(ch_plus_better_none = as.integer(trueval_ch_plus + tol < trueval_none)) %>%
#     mutate(ch_plus_better_ch = as.integer(trueval_ch_plus + tol < trueval_ch))
#
# df_trueval %>%
#     group_by(ch_better_none, ch_plus_better_none, ch_plus_better_ch) %>%
#     summarize(count = n())
#
# # Sanity checks
#
# # Calculate percentage of instances with predval_none > predval_ch (up to tolerance)
#
# tol <- 1.0e-4
#
# a <- df_wide$predval_none
# b <- df_wide$predval_ch
# tmp <- (a - b) / pmax(1, abs(0.5*(a + b)))
# sum(tmp > tol) / nrow(df_wide)
#
# # Calculate percentage of instances with predval_ch > predval_ch_plus (up to tolerance)
#
# a <- df_wide$predval_ch
# b <- df_wide$predval_ch_plus
# tmp <- (a - b) / pmax(1, abs(0.5*(a + b)))
# sum(tmp > tol) / nrow(df_wide)
#
# # Calculate percentage of instances with predval_svm > predval_ch (up to tolerance)
#
# # a <- df_wide$predval_svm
# # b <- df_wide$predval_ch
# # tmp <- (a - b) / pmax(1, abs(0.5*(a + b)))
# # sum(tmp > tol) / nrow(df_wide)
#
# # Calculate percentage of instances with predval_svm_plus > predval_ch_plus (up to tolerance)
#
# # a <- df_wide$predval_svm_plus
# # b <- df_wide$predval_ch_plus
# # tmp <- (a - b) / pmax(1, abs(0.5*(a + b)))
# # sum(tmp > tol) / nrow(df_wide)
#
# # Note: Autoencoder and Autoencoder+ are nonconvex
#
# # Statistics of improvement
#
# # Calculate percentage of times CH is better (no worse) than NONE in at least 1 of 3 measures
#
# a <- (df_wide$valerr_ch   <= df_wide$valerr_none)
# b <- (df_wide$distgmin_ch <= df_wide$distgmin_none)
# c <- (df_wide$distgval_ch <= df_wide$distgval_none)
# mean(a | b | c)
# hist(as.integer(a) + as.integer(b) + as.integer(c))
#
# # Calculate percentage of times CH_PLUS is better (no worse) than CH in at least 1 of 3 measures
#
# tmp <- df_wide
# tmp$a <- as.integer(df_wide$valerr_ch_plus   <= df_wide$valerr_ch)
# tmp$b <- as.integer(df_wide$distgmin_ch_plus <= df_wide$distgmin_ch)
# tmp$c <- as.integer(df_wide$distgval_ch_plus <= df_wide$distgval_ch)
# mean(tmp$a | tmp$b | tmp$c)
# # mean((1 - tmp$a) | (1 - tmp$b) | (1 - tmp$c))
# tmp$d <- tmp$a + tmp$b + tmp$c
# qplot(data = tmp, x = factor(d))
# tmp %>% group_by(a, b, c) %>% summarize(count = n(), d = unique(d)) %>%
#     arrange(d, a, b, c) %>% select(-d) %>%
#     rename("ch_plus_better_valerr" = a) %>%
#     rename("ch_plus_better_distgmin" = b) %>%
#     rename("ch_plus_better_distgval" = c)
#
# # Calculate percentage of times ch improves ch_plus (NOT PREFERRED!) in at least 1 of 3 measures
#
# a <- (df_wide$valerr_ch_plus   >= df_wide$valerr_ch)
# b <- (df_wide$distgmin_ch_plus >= df_wide$distgmin_ch)
# c <- (df_wide$distgval_ch_plus >= df_wide$distgval_ch)
# mean(a | b | c)
# hist(as.integer(a) + as.integer(b) + as.integer(c))
#
# # Calculate percentage of times ch_plus improves none in at least 1 of 3 measures
#
# a <- (df_wide$valerr_ch_plus   <= df_wide$valerr_none)
# b <- (df_wide$distgmin_ch_plus <= df_wide$distgmin_none)
# c <- (df_wide$distgval_ch_plus <= df_wide$distgval_none)
# mean(a | b | c)
# hist(as.integer(a) + as.integer(b) + as.integer(c))
#
# # Calculate percentage of times svm_plus improves svm in at least 1 of 3 measures
#
# # a <- (df_wide$valerr_svm_plus   <= df_wide$valerr_svm)
# # b <- (df_wide$distgmin_svm_plus <= df_wide$distgmin_svm)
# # c <- (df_wide$distgval_svm_plus <= df_wide$distgval_svm)
# # mean(a | b | c)
# # hist(as.integer(a) + as.integer(b) + as.integer(c))
#
# # tmp <- df_wide
# # tmp$a <- as.integer(df_wide$valerr_svm_plus   <= df_wide$valerr_svm)
# # tmp$b <- as.integer(df_wide$distgmin_svm_plus <= df_wide$distgmin_svm)
# # tmp$c <- as.integer(df_wide$distgval_svm_plus <= df_wide$distgval_svm)
# # # mean(a | b | c)
# # tmp$d <- tmp$a + tmp$b + tmp$c
# # qplot(data = tmp, x = factor(d))
# # tmp %>% group_by(a, b, c) %>% summarize(count = n(), d = unique(d)) %>%
# #     arrange(d, a, b, c) %>% select(-d) %>%
# #     rename("svm_plus_better_valerr" = a) %>%
# #     rename("svm_plus_better_distgmin" = b) %>%
# #     rename("svm_plus_better_distgval" = c)
#
# # Calculate percentage of times ae_plus improves ae in at least 1 of 3 measures
#
# # tmp <- df_wide
# # tmp$a <- as.integer(df_wide$valerr_ae_plus   <= df_wide$valerr_ae)
# # tmp$b <- as.integer(df_wide$distgmin_ae_plus <= df_wide$distgmin_ae)
# # tmp$c <- as.integer(df_wide$distgval_ae_plus <= df_wide$distgval_ae)
# # tmp$d <- tmp$a + tmp$b + tmp$c
# # qplot(data = tmp, x = factor(d))
# # tmp %>% group_by(a, b, c) %>% summarize(count = n(), d = unique(d)) %>%
# #     arrange(d, a, b, c) %>% select(-d) %>%
# #     rename("ae_plus_better_valerr" = a) %>%
# #     rename("ae_plus_better_distgmin" = b) %>%
# #     rename("ae_plus_better_distgval" = c)
#
# # Calculate percentage of times AE is better (no worse) than NONE in at least 1 of 3 measures
#
# # a <- (df_wide$valerr_ae   <= df_wide$valerr_none)
# # b <- (df_wide$distgmin_ae <= df_wide$distgmin_none)
# # c <- (df_wide$distgval_ae <= df_wide$distgval_none)
# # mean(a | b | c)
# # hist(as.integer(a) + as.integer(b) + as.integer(c))
#
# # Calculate percentage of times AE+ is better (no worse) than NONE in at least 1 of 3 measures
#
# # a <- (df_wide$valerr_ae_plus   <= df_wide$valerr_none)
# # b <- (df_wide$distgmin_ae_plus <= df_wide$distgmin_none)
# # c <- (df_wide$distgval_ae_plus <= df_wide$distgval_none)
# # mean(a | b | c)
# # hist(as.integer(a) + as.integer(b) + as.integer(c))
#
#
# # Pictures (for total error, not normalized total error)
#
# # tmp <- df %>%
# #     select(seed, func, sample_sz, noise_std, sampling, learning, v_domain, toterr) %>%
# #     pivot_wider(names_from = v_domain, values_from = toterr)
# #
# # qplot(data = tmp, x = ch_plus / ch, log = "x", facets = func ~ learning)
# #
# # qplot(data = tmp, x = svm_plus / svm, log = "x", facets = func ~ learning)
#
# # Examining distgmin, break and analyze different groups of improvement
#
# # df_wide <- df %>%
# #     select(
# #         seed, func, sample_sz, noise_std, sampling, learning, v_domain, distgmin
# #     ) %>%
# #     pivot_wider(names_from = v_domain, values_from = distgmin)
# #
# # tol <- 1.0e-4
# #
# # tmp <- df_wide %>%
# #     mutate(ch_plus = ch_plus / ch, svm_plus = svm_plus / svm) %>%
# #     select(-ch, -svm) %>%
# #     mutate(ch_plus_distgmin = ifelse(ch_plus < 1 - tol, "better", ifelse(ch_plus > 1 + tol, "worse", "same"))) %>%
# #     mutate(svm_plus_distgmin = ifelse(svm_plus < 1 - tol, "better", ifelse(svm_plus > 1 + tol, "worse", "same")))
# #
# # tmp %>%
# #     group_by(ch_plus_distgmin) %>%
# #     summarize(count = n(), median_ratio = median(ch_plus), mean_ratio = mean(ch_plus))
# #
# # tmp %>%
# #     group_by(svm_plus_distgmin) %>%
# #     summarize(count = n(), med_change = median(svm_plus), mean_ratio = mean(svm_plus))
#
# # Examining valerr, break and analyze different groups of improvement
#
# df_wide <- df %>%
#     select(
#         seed, func, sample_sz, noise_std, sampling, learning, v_domain, valerr
#     ) %>%
#     pivot_wider(names_from = v_domain, values_from = valerr)
#
# tol <- 1.0e-4
#
# # tmp <- df_wide %>%
# #     mutate(ch_plus = ch_plus / ch, svm_plus = svm_plus / svm) %>%
# #     select(-ch, -svm) %>%
# #     mutate(ch_plus_valerr = ifelse(ch_plus < 1 - tol, "better", ifelse(ch_plus > 1 + tol, "worse", "same"))) %>%
# #     mutate(svm_plus_valerr = ifelse(svm_plus < 1 - tol, "better", ifelse(svm_plus > 1 + tol, "worse", "same")))
#
# tmp <- df_wide %>%
#     mutate(ch_plus = ch_plus / ch) %>%
#     select(-ch) %>%
#     mutate(ch_plus_valerr = ifelse(ch_plus < 1 - tol, "better", ifelse(ch_plus > 1 + tol, "worse", "same")))
#
# tmp %>%
#     group_by(ch_plus_valerr) %>%
#     summarize(count = n(), median_ratio = median(ch_plus), mean_ratio = mean(ch_plus))
#
# # tmp %>%
# #     group_by(svm_plus_valerr) %>%
# #     summarize(count = n(), med_change = median(svm_plus), mean_ratio = mean(svm_plus))
#
# # Examining distgval, break and analyze different groups of improvement
#
# # df_wide <- df %>%
# #     select(
# #         seed, func, sample_sz, noise_std, sampling, learning, v_domain, distgval
# #     ) %>%
# #     pivot_wider(names_from = v_domain, values_from = distgval)
# #
# # tol <- 1.0e-4
# #
# # tmp <- df_wide %>%
# #     mutate(ch_plus = ch_plus / ch, svm_plus = svm_plus / svm) %>%
# #     select(-ch, -svm) %>%
# #     mutate(ch_plus_distgval = ifelse(ch_plus < 1 - tol, "better", ifelse(ch_plus > 1 + tol, "worse", "same"))) %>%
# #     mutate(svm_plus_distgval = ifelse(svm_plus < 1 - tol, "better", ifelse(svm_plus > 1 + tol, "worse", "same")))
# #
# # tmp %>%
# #     group_by(ch_plus_distgval) %>%
# #     summarize(count = n(), median_ratio = median(ch_plus), mean_ratio = mean(ch_plus))
# #
# # tmp %>%
# #     group_by(svm_plus_distgval) %>%
# #     summarize(count = n(), med_change = median(svm_plus), mean_ratio = mean(svm_plus))
#
#
# ################################################################################
#
#
#
# # df <- df %>%
# #     arrange(func, sample_sz, noise_std)
# #
# # tmp <- df %>%
# #     group_by(validity_domain, noise_std) %>%
# #     summarize(mean_total_err = mean(total_error)) %>%
# #     arrange(noise_std)
# #
# # #tmp <- df %>%
# # #    group_by(validity_domain, sample_size) %>%
# # #    summarize(
# # #        mean_pred_err_in_std = mean(prediction_error_in_std)
# # #    ) %>%
# # #    arrange(sample_size)
# #
# # # p <- qplot(data = df, x = prediction_error_in_std, y = normalized_distance_to_global_min, log = "x", color = validity_domain)
# # # print(p)
# #
# # tmp <- df %>%
# #     select(seed, func, sample_sz, noise_std, sampling, learning, v_domain, total_error) %>%
# #     pivot_wider(names_from = v_domain, values_from = total_error) %>%
# #     clean_names() %>%
# #     mutate(log10ratio_ch = log10(ch_2 / ch), log10ratio_svm = log10(pwl_kernel_svm_2 / pwl_kernel_svm)) %>%
# #     ungroup() %>%
# #     group_by(learning, func) %>%
# #     summarize(n = n(),
# #               medlog10ratio_ch = median(log10ratio_ch), meanlog10ratio_ch = mean(log10ratio_ch),
# #               medlog10ratio_svm = median(log10ratio_svm), meanlog10ratio_svm = mean(log10ratio_svm)
# #               ) %>%
# #     mutate(medratio_ch = 10^medlog10ratio_ch, meanratio_ch = 10^meanlog10ratio_ch,
# #            medratio_svm = 10^medlog10ratio_svm, meanratio_svm = 10^meanlog10ratio_svm)
# #
# # tmp <- df %>%
# #     select(seed, func, sample_sz, noise_std, sampling, learning, v_domain, total_error) %>%
# #     pivot_wider(names_from = v_domain, values_from = total_error) %>%
# #     clean_names()
# #
# # qplot(data = tmp, x = ch_2 / ch, log = "x", facets = func ~ learning)
# #
# # qplot(data = tmp, x = pwl_kernel_svm_2 / pwl_kernel_svm, log = "x", facets = func ~ learning)
# #
# # ###
# #
# # df_wide <- df %>%
# #     select(
# #         seed, func, sample_sz, noise_std, sampling, learning, v_domain,
# #         percent_in_ch, normalized_total_error
# #         ) %>%
# #     pivot_wider(names_from = v_domain, values_from = normalized_total_error) %>%
# #     clean_names() %>%
# #     mutate(ratio_ch = ch_2 / ch) %>%
# #     mutate(ratio_svm = pwl_kernel_svm_2 / pwl_kernel_svm)
# #
# #
# # ###
# #
# # df_wide <- df %>%
# #     select(
# #         seed, func, sample_sz, noise_std, sampling, learning, v_domain,
# #         trueval
# #     ) %>%
# #     pivot_wider(names_from = v_domain, values_from = c("trueval"))
