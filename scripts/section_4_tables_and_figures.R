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

# Read data from output.csv

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

# Filter as necessary. As of 2024-05-23, expected number of 
# rows in df is 37800 * 4 = 151200

# Remove rastrigin2d instances
# Remove bestsample v-domain
# Remove svm v-domain
# Remove svmplus v-domain

df <- df %>% filter(func != "rastrigin2d")
df <- df %>% filter(v_domain != "bestsample")
df <- df %>% filter(v_domain != "svm")
df <- df %>% filter(v_domain != "svmplus")

# Sort (not really necessary, just for cleanliness)

df <- df %>% arrange(seed, func, noise, v_domain)

# Factor v_domain column

df$v_domain <- factor(df$v_domain, c("box", "ch", "isofor", "chplus"))

# Create plot function (as of 2024-05-23, this is not being used in this file)

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
    # tmp[[2]] <- formatC(tmp[[2]] / val, format = "f", digits = 2)
    
    val <- tmp$med_optval_err[1] # box
    df$optval_err <- df$optval_err / val
    # tmp[[3]] <- formatC(tmp[[3]] / val, format = "f", digits = 2)

    val <- tmp$med_optsol_err[1] # box
    df$optsol_err <- df$optsol_err / val
    # tmp[[4]] <- formatC(tmp[[4]] / val, format = "f", digits = 2)

    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "box"] <- "Box"
    tmp$v_domain[tmp$v_domain == "ch"] <- "CH"
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    tmp$v_domain[tmp$v_domain == "isofor"] <- "IsoFor"
    names(tmp) <- c("Validity\nDomain", "Median Function\nValue Error", "Median Optimal\nValue Error", "Median Optimal\nSolution Error")

    print(tmp %>% xtable(digits = 2))
    
    # fname <- paste0("../results/plots/", my_func, "_", my_sampling, "_", my_learning, "_table.png")
    # png(fname, res = 300, height = 200*nrow(tmp), width = 400*ncol(tmp))
    # tmp <- apply(tmp, 1:2, formatC, format="f", digits=3)
    # grid.table(tmp)
    # dev.off()
   
    # print(my_func)
    # print(my_sampling)
    # print(my_learning)
    # df <- filter(df, func == my_func, sampling == my_sampling, learning == my_learning)
    # df$v_domain <- as.character(df$v_domain)
    # df$v_domain[df$v_domain == "box"] <- "Box"
    # df$v_domain[df$v_domain == "ch"] <- "CH"
    # df$v_domain[df$v_domain == "chplus"] <- "CH+"
    # df$v_domain[df$v_domain == "isofor"] <- "IsoFor"
    # df$v_domain <- factor(df$v_domain, c("Box", "CH", "IsoFor", "CH+"))
    # plot1 <- ggplot(df, aes(x = funval_err, y = fct_rev(v_domain), fill = v_domain)) +
    #     stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
    #     scale_x_continuous(trans = "log10", n.breaks = 4, labels = number) +
    #     theme(legend.position = "none") +
    #     ggtitle("Function Value Error") +
    #     theme(axis.title = element_blank()) +
    #     theme(axis.ticks = element_blank()) +
    #     theme(axis.text.y = element_blank())
    # plot2 <- ggplot(df, aes(x = optval_err, y = fct_rev(v_domain), fill = v_domain)) +
    #     stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
    #     scale_x_continuous(trans = "log10", n.breaks = 4, labels = number) +
    #     theme(legend.position = "none") +
    #     ggtitle("Optimal Value Error") +
    #     theme(axis.title = element_blank()) +
    #     theme(axis.ticks = element_blank()) +
    #     theme(axis.text.y = element_blank())
    # plot3 <- ggplot(df, aes(x = optsol_err, y = fct_rev(v_domain), fill = v_domain)) +
    #     stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
    #     scale_x_continuous(trans = "log10", n.breaks = 4, labels = number) +
    #     ggtitle("Optimal Solution Error") +
    #     theme(axis.title = element_blank()) +
    #     theme(axis.ticks = element_blank()) +
    #     theme(axis.text.y = element_blank()) +
    #     scale_fill_hue(name = "Validity\nDomain")
    #
    # p <- grid.arrange(plot1, plot2, plot3, nrow = 1, widths = c(1, 1, 1.3))
    # fname <- paste0("../results/plots/", my_func, "_", my_sampling, "_", my_learning, ".png")
    # ggsave(fname, p, width = 9, height = 4, dpi = 300)

}

# Create table printing function

my_print_function <- function(my_sampling) {
    
    tmp <- df %>%
        filter(sampling == my_sampling) %>%
        select(
            func, noise, sampling, sample_sz, seed, learning, v_domain,
            funval_err, optval_err, optsol_err
        ) %>%
        group_by(func, v_domain) %>%
        summarize(med_funval_err = median(funval_err),
                  med_optval_err = median(optval_err),
                  med_optsol_err = median(optsol_err))
    
    # Be careful with this code! Is there a better way to do this?
    
    for(j in 1 : (nrow(tmp)/4)) {
        
        i <- (j - 1)*4 + 1
        
        val <- tmp$med_funval_err[i] # box
        tmp$med_funval_err[i + 0] <- tmp$med_funval_err[i + 0] / val
        tmp$med_funval_err[i + 1] <- tmp$med_funval_err[i + 1] / val
        tmp$med_funval_err[i + 2] <- tmp$med_funval_err[i + 2] / val
        tmp$med_funval_err[i + 3] <- tmp$med_funval_err[i + 3] / val
        
        val <- tmp$med_optval_err[i] # box
        tmp$med_optval_err[i + 0] <- tmp$med_optval_err[i + 0] / val
        tmp$med_optval_err[i + 1] <- tmp$med_optval_err[i + 1] / val
        tmp$med_optval_err[i + 2] <- tmp$med_optval_err[i + 2] / val
        tmp$med_optval_err[i + 3] <- tmp$med_optval_err[i + 3] / val
        
        val <- tmp$med_optsol_err[i] # box
        tmp$med_optsol_err[i + 0] <- tmp$med_optsol_err[i + 0] / val
        tmp$med_optsol_err[i + 1] <- tmp$med_optsol_err[i + 1] / val
        tmp$med_optsol_err[i + 2] <- tmp$med_optsol_err[i + 2] / val
        tmp$med_optsol_err[i + 3] <- tmp$med_optsol_err[i + 3] / val
        
    }
    
    tmp$med_funval_err <- formatC(tmp$med_funval_err, format = "f", digits = 2)
    tmp$med_optval_err <- formatC(tmp$med_optval_err, format = "f", digits = 2)
    tmp$med_optsol_err <- formatC(tmp$med_optsol_err, format = "f", digits = 2)
    
    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "box"] <- "Box"
    tmp$v_domain[tmp$v_domain == "ch"] <- "CH"
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    tmp$v_domain[tmp$v_domain == "isofor"] <- "IsoFor"
    names(tmp) <- c("Func", "V Dom", "Med FunValErr", "Med OptValErr", "Med OptSolErr")
    
    names(tmp) <- c("Function", "Validity Domain", "Median Function Value Error",
                    "Median Optimval Value Error", "Median Optimal Solution Error")
    
    tmp$Function[tmp$Function == "beale"] <- "Beale"
    tmp$Function[tmp$Function == "griewank"] <- "Griewank"
    tmp$Function[tmp$Function == "peaks"] <- "Peaks"
    tmp$Function[tmp$Function == "powell"] <- "Powell"
    tmp$Function[tmp$Function == "qing"] <- "Qing"
    tmp$Function[tmp$Function == "quintic"] <- "Quintic"
    tmp$Function[tmp$Function == "rastrigin10d"] <- "Rastrigin"
    
    tmp$`Validity Domain`[tmp$`Validity Domain` == "Box"] <- "{\\sc Box}"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "IsoFor"] <- "{\\sc IsoFor}"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "CH+"] <- "$\\CH^+$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "CH"] <- "$\\CH$"
    
    # print(
    #     tmp %>% xtable(digits = 2),
    #     include.rownames = FALSE,
    #     sanitize.text.function = function(x){x},
    #     file = paste0("../results/tables/", my_sampling, ".tex")
    #     )
    
    tmp
}

# Save Tables XXX and YYY (2024-05-23) to file. These tables show how we perform
# on the various functions. Table XXX shows the uniform sampling rule. Table YYY
# shows the normal_at_min sampling rule.

df_uniform <- my_print_function(my_sampling = "uniform")
df_normal <- my_print_function(my_sampling = "normal_at_min")
#my_print_function(my_sampling = "all")

# 2024-06-12: Attempt to combine uniform and normal_at_min tables

df_all <- cbind(
    df_uniform[, c(1, 2, 3)],
    df_normal[, 3],
    df_uniform[, 4],
    df_normal[, 4],
    df_uniform[, 5],
    df_normal[, 5]
)
print(
    df_all %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/all.tex"
)

# Save table showing R2 scores, similar to table 1 in Shi et al.

tmp <- df %>%
    filter(v_domain == "box") %>%
    group_by(func, learning) %>%
    summarize(avgR2 = mean(R2_score)) %>%
    pivot_wider(names_from = func, values_from = avgR2)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(Beale = beale) %>%
    rename(Griewank = griewank) %>%
    rename(Peaks = peaks) %>%
    rename(Powell = powell) %>%
    rename(Qing = qing) %>%
    rename(Quintic = quintic) %>%
    rename(Rastrigin = rastrigin10d) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/r2.tex"
)

# Save table showing median ML training times (similarly constructed as table for R2 scores)

tmp <- df %>%
    filter(v_domain == "box") %>%
    group_by(func, learning) %>%
    summarize(medtime = median(ML_train_time)) %>%
    pivot_wider(names_from = func, values_from = medtime)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(Beale = beale) %>%
    rename(Griewank = griewank) %>%
    rename(Peaks = peaks) %>%
    rename(Powell = powell) %>%
    rename(Qing = qing) %>%
    rename(Quintic = quintic) %>%
    rename(Rastrigin = rastrigin10d) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/ml_train_times.tex"
)

# Save table showing median opt setup times

tmp <- df %>%
    group_by(v_domain, learning) %>%
    summarize(myval = median(opt_setup_time)) %>%
    pivot_wider(names_from = v_domain, values_from = myval)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(`{\\sc Box}` = box) %>%
    rename(`$\\CH$` = ch) %>%
    rename(`{\\sc IsoFor}` = isofor) %>%
    rename(`$\\CH^+$` = chplus) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/opt_setup_times.tex"
)

# Save table showing median opt solve times (similarly constructed as table for setup times)

tmp <- df %>%
    group_by(v_domain, learning) %>%
    summarize(myval = median(opt_opt_time)) %>%
    pivot_wider(names_from = v_domain, values_from = myval)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(`{\\sc Box}` = box) %>%
    rename(`$\\CH$` = ch) %>%
    rename(`{\\sc IsoFor}` = isofor) %>%
    rename(`$\\CH^+$` = chplus) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/opt_solve_times.tex"
)


# Figures

# tmp <- df %>% select(-modval_modsol, -truval_modsol, -optsol_err, -optval_err,
#                      -opt_setup_time, -opt_opt_time, -ML_train_time, -R2_score)
# tmp <- tmp %>% pivot_wider(names_from = v_domain, values_from = funval_err)
# p <- qplot(data = tmp, x = box, y = chplus, log = "xy", geom = "density2d")
# mylims <- range(with(tmp, c(box, chplus)))
# p <- p + coord_cartesian(xlim = mylims, ylim = mylims)
# p <- p + annotate("segment", x = mylims[1], xend = mylims[2], y = mylims[1], yend = mylims[2])
# p <- p + annotate("text", x = 1.0e+2, y = 1.0e-3, label = paste(sum(tmp$chplus < tmp$box), "experiments"))
# p <- p + annotate("text", x = 1.0e-3, y = 1.0e+2, label = paste(nrow(tmp) - sum(tmp$chplus < tmp$box), "experiments"))
#print(p)
# fname <- "scratch_box.png"
# ggsave(fname, p, width = 4, height = 4, dpi = 300)


# tmp <- df %>% select(-modval_modsol, -truval_modsol, -optsol_err, -optval_err,
#                      -opt_setup_time, -opt_opt_time, -ML_train_time, -R2_score)
# tmp <- tmp %>% pivot_wider(names_from = v_domain, values_from = funval_err)
# p <- qplot(data = tmp, x = ch, y = chplus, log = "xy", geom = "density2d")
# mylims <- range(with(tmp, c(ch, chplus)))
# p <- p + coord_cartesian(xlim = mylims, ylim = mylims)
# p <- p + annotate("segment", x = mylims[1], xend = mylims[2], y = mylims[1], yend = mylims[2])
# p <- p + annotate("text", x = 1.0e+2, y = 1.0e-3, label = paste(sum(tmp$chplus < tmp$ch), "experiments"))
# p <- p + annotate("text", x = 1.0e-3, y = 1.0e+2, label = paste(nrow(tmp) - sum(tmp$chplus < tmp$ch), "experiments"))
# print(p)
# fname <- "scratch_ch.png"
# ggsave(fname, p, width = 4, height = 4, dpi = 300)

# Timings

# qplot(data = df, x = opt_setup_time, log = "x", fill = learning)
# qplot(data = df, x = opt_opt_time, log = "x", fill = learning, geom = "density", alpha = I(0.5))
#
# qplot(data = df, x = opt_setup_time, log = "x", fill = v_domain)
# qplot(data = df, x = opt_opt_time, log = "x", fill = v_domain, geom = "density", alpha = I(0.5))
#
# qplot(data = (df %>% filter(v_domain == "box")), x = ML_train_time, log = "x", fill = learning)
#
# tmp <- filter(df, func == "powell")
# qplot(data = tmp, x = v_domain, y = opt_opt_time + 1, geom = "boxplot", color = learning, log = "y")
