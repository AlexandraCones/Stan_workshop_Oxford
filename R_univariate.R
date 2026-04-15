# =================================== set up house ================================================== #

library(cmdstanr)
library(rstan)
library(shinystan)
library(reshape2)
library(tidybayes)
library(bayesplot)
library(posterior)
library(ggplot2)
library(tidyverse)
library(ggdist)

# =================================== compile the model ================================================== #

set_cmdstan_path("/Users/alexcones/.cmdstan/cmdstan-2.36.0")
univariate_model = cmdstan_model(stan_file = "stan_univariate.stan", stanc_options = list("O1"))

# =============================== input and format the data ============================================== #

data_frame <- read.csv("simulated_data_z.csv", header = T) # input data
data_frame$ID <- as.integer(factor(data_frame$individual)) # create index of individual id values
data_frame$x <- as.numeric(scale(data_frame$environment, scale = TRUE)) # scale x variable
data_frame$z <- as.numeric(scale(data_frame$y1, scale = TRUE)) # scale response variable

# ============================= create the data list for Stan ============================================ #

stan_data_frame <- list(
  N_z = nrow(data_frame),
  N_ID = length(unique(data_frame$ID)),  
  ID = data_frame$ID,               
  x = data_frame$x,           
  z = data_frame$z)

# ====================================== run the model =================================================== #

univariate_model_fit <- univariate_model$sample(
  data = stan_data_frame,      
  chains = 8,        
  parallel_chains = 8,      
  iter_sampling = 3000,         
  iter_warmup = 1000,            
  adapt_delta = 0.95)

# ==================================== check the model =================================================== #

draws <- univariate_model_fit$draws()
parameters <- c("beta_0", "beta_1", "Var_alpha_0", "Var_alpha_1", "Cov_alpha_01", "sigma_0")

# divergences, treedepth, ebfmi
univariate_model_fit$diagnostic_summary()

# rhat, ESS
univariate_model_fit$summary(parameters) |> select(variable, rhat, ess_bulk, ess_tail)

# trace plots
mcmc_trace(draws, parameters)

# posterior distributions
mcmc_dens_overlay(draws,parameters)

# pair plots
mcmc_pairs(draws, parameters)

# ===================================== get results ====================================================== #

univariate_model_fit$summary(parameters)

# ===================================== plot results ====================================================== #

# fixed effects
fixed_parameters <- c("beta_0", "beta_1")
fixed_draws <- univariate_model_fit$draws(fixed_parameters, format = "df") %>%
  pivot_longer(cols = all_of(fixed_parameters), names_to = "variable", values_to = "value")
fixed_fit <- univariate_model_fit$summary(fixed_parameters)

ggplot() +
  stat_halfeye(data = fixed_draws, aes(x = value, y = variable), 
               .width = c(0.05, 0.95), fill = "pink", alpha = 0.6) +
  geom_pointrange(data = fixed_fit, aes(x = median, y = variable, xmin = q5, xmax = q95)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  labs(x = "Posterior estimate", y = "Parameter") +
  theme_classic(base_size = 18) +
  theme(
    axis.text.y = element_text(size = 12),
    axis.line = element_line(linewidth = 1.2),
    axis.title = element_text(size = 20, face = "bold"))

# plot random effects
random_parameters <- c("Var_alpha_0", "Var_alpha_1", "Cov_alpha_01")
random_draws <- univariate_model_fit$draws(random_parameters, format = "df") %>%
  pivot_longer(cols = all_of(random_parameters), names_to = "variable", values_to = "value")
random_fit <- univariate_model_fit$summary(random_parameters)

ggplot() +
  stat_halfeye(data = random_draws, aes(x = value, y = variable), 
               .width = c(0.05, 0.95), fill = "pink", alpha = 0.6) +
  geom_pointrange(data = random_fit, aes(x = median, y = variable, xmin = q5, xmax = q95)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  labs(x = "Posterior estimate", y = "Parameter") +
  theme_classic(base_size = 18) +
  theme(
    axis.text.y = element_text(size = 12),
    axis.line = element_line(linewidth = 1.2),
    axis.title = element_text(size = 20, face = "bold"))

