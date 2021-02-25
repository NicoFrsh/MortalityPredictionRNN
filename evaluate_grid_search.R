# This script evaluates the models created by the grid search.

library(tidyverse)
library(dplyr)
library(keras)
library(tfruns)
library(data.table)


source("out_of_sample_calculations.R")
source("create_model.R")

# Verbose parameters
calculate_loss <- FALSE
plot_grid <- FALSE
show_projections <- TRUE

# model type to be evaluated ("LSTM" or "GRU")
model_type <- "LSTM"

# Number of models used for the calculation of the out of sample loss.
number_of_models <- 10
# Index of model to be used to project the future rates.
model_rank <- 1

# fixed parameters
last_observed_year <- 2006
countries <- c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")

# Load data.
data <- fread("./data/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% countries),]

results <- ls_runs(order = metric_val_loss, decreasing= F, runs_dir = paste0('grid_search_', model_type))
results <- select(results, -c(output))

# Write to xlsx.
writexl::write_xlsx(results, paste0("./data/results_", model_type,".xlsx"))

# Load the models to be used for the predictions.
models <- c()
timesteps <- c()
age_ranges <- c()
for (model_index in 1:number_of_models) {
        unit_sizes <- c(results[model_index,11], results[model_index,12], results[model_index,13], results[model_index,14], results[model_index,15])
        unit_sizes <- unit_sizes[1:results[model_index,10]]
        if (results[model_index,7] == "LSTM") {
                model <- create_lstm_model(c(results[model_index,8], results[model_index,9]), unit_sizes, results[model_index,17], results[model_index,18], results[model_index,19], 0)
        }
        else {
                model <- create_gru_model(c(results[model_index,8], results[model_index,9]), unit_sizes, results[model_index,17], results[model_index,18], results[model_index,19], 0)
        }
        load_model_weights_hdf5(model, paste0("./", results[model_index,1], "/best_model_weights.h5"))

        # Add models and parameters of the dataset to their corresponding lists.
        models <- c(models, model)
        timesteps <- c(timesteps, results[model_index, 8])
        age_ranges <- c(age_ranges, results[model_index, 9])
}

if (calculate_loss){
        table <- out_of_sample_loss(models, data, countries, timesteps, age_ranges, last_observed_year)
}


if (show_projections){
        if (model_rank <= number_of_models){
                # Project mortality rates
                future_rates <- project_future_rates(models[[model_rank]], data, countries, results[model_rank,8], results[model_rank,9], 
                                                     last_observed_year, 40)
                
        } else {
                unit_sizes <- c(results[model_rank,11], results[model_rank,12], results[model_rank,13], results[model_rank,14], results[model_rank,15])
                unit_sizes <- unit_sizes[1:results[model_rank,10]]
                if (results[model_rank,7] == "LSTM") {
                        model <- create_lstm_model(c(results[model_rank,8], results[model_rank,9]), unit_sizes, results[model_rank,17], 
                                                   results[model_rank,18], results[model_rank,19], 0)
                }
                else {
                        model <- create_gru_model(c(results[model_rank,8], results[model_rank,9]), unit_sizes, results[model_rank,17], 
                                                  results[model_rank,18], results[model_rank,19], 0)
                }
                load_model_weights_hdf5(model, paste0("./", results[model_rank,1], "/best_model_weights.h5"))
                
                future_rates <- project_future_rates(model, data, countries, results[model_rank,8], results[model_rank,9], last_observed_year, 40)
        }
        
        future_rates_DEUT_female <- future_rates[[get_country_index("DEUT", countries) + 1]]$Female_Pred
        future_rates_DEUT_male <- future_rates[[get_country_index("DEUT", countries) + 1]]$Male_Pred
        
        plot_DEUT_Female <- ggplot() +
                ggtitle("DEUT Female") +
                geom_point(data = future_rates_DEUT_female[which(future_rates_DEUT_female$Year == 2006)], aes(x = Age, y = log_mortality, color = "Year 2006")) +
                geom_point(data = future_rates_DEUT_female[which(future_rates_DEUT_female$Year == 2016)], aes(x = Age, y = log_mortality, color = "Year 2016")) +
                geom_point(data = future_rates_DEUT_female[which(future_rates_DEUT_female$Year == 2026)], aes(x = Age, y = log_mortality, color = "Year 2026")) +
                geom_point(data = future_rates_DEUT_female[which(future_rates_DEUT_female$Year == 2036)], aes(x = Age, y = log_mortality, color = "Year 2036")) +
                geom_point(data = future_rates_DEUT_female[which(future_rates_DEUT_female$Year == 2046)], aes(x = Age, y = log_mortality, color = "Year 2046"))
        
        plot_DEUT_Male <- ggplot() +
                ggtitle("DEUT Male") +
                geom_point(data = future_rates_DEUT_male[which(future_rates_DEUT_male$Year == 2006)], aes(x = Age, y = log_mortality, color = "Year 2006")) +
                geom_point(data = future_rates_DEUT_male[which(future_rates_DEUT_male$Year == 2016)], aes(x = Age, y = log_mortality, color = "Year 2016")) +
                geom_point(data = future_rates_DEUT_male[which(future_rates_DEUT_male$Year == 2026)], aes(x = Age, y = log_mortality, color = "Year 2026")) +
                geom_point(data = future_rates_DEUT_male[which(future_rates_DEUT_male$Year == 2036)], aes(x = Age, y = log_mortality, color = "Year 2036")) +
                geom_point(data = future_rates_DEUT_male[which(future_rates_DEUT_male$Year == 2046)], aes(x = Age, y = log_mortality, color = "Year 2046"))
        
        print(plot_DEUT_Female)
        print(plot_DEUT_Male)
        
        
}

#  plot observed vs fitted mortality for all country-gender combinations
if (plot_grid & show_projections){
        pred_year <- 2014
        
        library(gridExtra)
        
        plot_list <- vector("list", 18)
        
        counter <- 1
        for (country in countries){
                future_rates_female <- future_rates[[get_country_index(country, countries) + 1]]$Female_Pred
                future_rates_male <- future_rates[[get_country_index(country, countries) + 1]]$Male_Pred
                
                p_female <- ggplot() +
                        ggtitle(paste0(country, " Female")) +
                        geom_point(data = data[which(data$Year == pred_year & data$Gender == "Female" & data$Country == country)],
                                   aes(x = Age, y = log_mortality), color = "red", size = 0.5) +
                        geom_line(data = future_rates_female[which(future_rates_female$Year == pred_year)], 
                                  aes(x = Age, y = log_mortality),  color = "blue") +
                        theme(plot.title = element_text(color = "black", size=8, hjust = 0.5,face="bold"))
                
                p_male <- ggplot() +
                        ggtitle(paste0(country, " Male")) +
                        geom_point(data = data[which(data$Year == pred_year & data$Gender == "Male" & data$Country == country)],
                                   aes(x = Age, y = log_mortality), color = "red", size = 0.5) +
                        geom_line(data = future_rates_male[which(future_rates_male$Year == pred_year)], 
                                  aes(x = Age, y = log_mortality), color = "blue") +
                        theme(plot.title = element_text(color = "black", size=8, hjust = 0.5,face="bold"))
                
                
                plot_list[[counter]] <- p_female
                plot_list[[counter+1]] <- p_male
                
                counter <- counter + 2
        }
        
        grid_plot <- grid.arrange(grobs = plot_list, nrow = 3)
        
        #ggsave(filename = "./plots/predictGrid.png", plot = grid_plot, dpi = 400, scale = 2)

}