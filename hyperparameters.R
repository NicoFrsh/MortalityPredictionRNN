# This R Script serves as Input for the 'tuning_run()'-function which is called in rnn_model.R
# and specifies the flags for the construction of various models from which we want to choose 
# the best performing one.
library(keras)
library(tensorflow)
#tensorflow::tf$random$set_seed(10)
#use_session_with_seed(10, disable_gpu = TRUE, disable_parallel_cpu = FALSE, quiet = FALSE)

source("create_model.R")


FLAGS <- flags(
        flag_string('model_type', 'LSTM'),
        flag_integer('timesteps', 10),
        flag_integer('age_range', 5),
        flag_integer('layers', 3),
        flag_integer('feature_dimension0', 20),
        flag_integer('feature_dimension1', 15),
        flag_integer('feature_dimension2', 10),
        flag_integer('feature_dimension3', 5),
        flag_integer('feature_dimension4', 2),
        #flag_numeric('dropout', 0.05),
        #flag_numeric('lr', 0.01),
        #flag_integer('patience', 35),
        #flag_integer('pats', 20),
        flag_integer('batch_size', 100),
        flag_string('activation', 'tanh'),
        flag_string('recurrent_activation', 'tanh'),
        flag_string("output_activation", "exponential")
)

# Data Preparation
# read RDS file
combined_training_set <- readRDS(file = paste0("./data/training_data/training_set_", FLAGS$timesteps, "_", FLAGS$age_range, ".rds"))
x_train <- combined_training_set[[1]]
y_train <- combined_training_set[[2]]
rm(combined_training_set)

# The mean of y_train will be used as starting value for the intercept weight as it leeds to 
# faster convergence.
average_label <- mean(y_train)


# construct unit sizes vector
unit_sizes <- c(FLAGS$feature_dimension0, FLAGS$feature_dimension1, FLAGS$feature_dimension2, FLAGS$feature_dimension3, FLAGS$feature_dimension4)
unit_sizes <- unit_sizes[1:FLAGS$layers]

# Create the wanted model.
if (FLAGS$model_type == "LSTM") {
        model <- create_lstm_model(c(FLAGS$timesteps, FLAGS$age_range), unit_sizes, FLAGS$activation, FLAGS$recurrent_activation, FLAGS$output_activation, average_label)
} else
        model <- create_gru_model(c(FLAGS$timesteps, FLAGS$age_range), unit_sizes, FLAGS$activation, FLAGS$recurrent_activation, FLAGS$output_activation, average_label)


# compile model
optimizer <- optimizer_adam()
model %>% compile(optimizer = optimizer, loss = "mse", metrics = list("mae"))

# early stopping
early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 35)

# adaptive learning rate
lr_reducer <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                            patience = 25, verbose = 0, mode = "min",
                                            min_delta = 1e-03, cooldown = 0, min_lr = 0)

# use best model only
use_best_model <- callback_model_checkpoint("best_model_weights.h5", monitor = "val_loss", verbose = 1,
                                            save_best_only = TRUE, save_weights_only = TRUE)

# fit model
history <- model %>% fit(x = x_train, y = y_train, validation_split = 0.1, batch_size = FLAGS$batch_size, epochs = 200, verbose = 1,
                         callbacks = list(early_stop, lr_reducer, use_best_model))

