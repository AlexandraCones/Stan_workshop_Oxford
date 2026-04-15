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
eiv_model = cmdstan_model(stan_file = "stan_eiv.stan", stanc_options = list("O1"))

# =============================== input and format the data ============================================== #

data_frame_z <- read.csv("tryout_data_z.csv", header = T) # input data
data_frame_w <- read.csv("tryout_data_w.csv", header = T) # input data
data_frame_z$ID <- as.integer(factor(data_frame_z$individual)) # create index of individual id values
data_frame_w$ID <- as.integer(factor(data_frame_w$individual)) # create index of individual id values
data_frame_z$x <- as.numeric(scale(data_frame_z$x1, scale = TRUE)) # scale x variable
data_frame_z$z1 <- as.numeric(scale(data_frame_z$trait1, scale = TRUE)) # scale response variable
data_frame_z$z2 <- as.numeric(scale(data_frame_z$trait2, scale = TRUE)) # scale response variable
data_frame_w$w <- as.numeric(scale(data_frame_w$fitness, scale = TRUE)) # scale response variable

# ============================= create the data list for Stan ============================================ #

stan_data_frame <- list(
  N_z = nrow(data_frame_z),
  N_w = nrow(data_frame_w),
  N_ID = length(unique(data_frame_z$ID)),  
  ID_z = data_frame_z$ID,
  ID_w = data_frame_w$ID,  
  x = data_frame_z$x,           
  z1 = data_frame_z$z1,
  z2 = data_frame_z$z2,
  w = data_frame_w$w)

# ====================================== run the model =================================================== #

tryout_model_fit <- tryout_model$sample(
  data = stan_data_frame,      
  chains = 8,        
  parallel_chains = 8,      
  iter_sampling = 500,         
  iter_warmup = 1000,            
  adapt_delta = 0.95)

# ==================================== check the model =================================================== #

draws <- tryout_model_fit$draws()
parameters <- c("beta_0", "beta_1", "beta_0_w", "beta_1_w", "beta_2_w", "beta_3_w",
                "Var_alpha_0_z1", "Var_alpha_1_z1", "Var_alpha_0_z2", "Var_alpha_1_z2",
                "Rho_int_z1_slope_z1", "Rho_int_z2_slope_z2", "Rho_int_z1_int_z2", "Rho_slope_z1_slope_z2",
                "Cov_alpha_0_z1_alpha_1_z1", "Cov_alpha_0_z2_alpha_1_z2")

# divergences, treedepth, ebfmi
tryout_model_fit$diagnostic_summary()

# rhat, ESS
tryout_model_fit$summary(parameters) |> select(variable, rhat, ess_bulk, ess_tail)

# trace plots
mcmc_trace(draws, parameters)

# posterior distributions
mcmc_dens_overlay(draws,parameters)

# pair plots
mcmc_pairs(draws, parameters)

# ===================================== get results ====================================================== #

tryout_model_fit$summary(parameters)

# ===================================== plot results ====================================================== #


# Rho_slope_z1_slope_z2 
# beta_3_w

# fixed effects
fixed_parameters <- c("beta_0", "beta_1", "beta_0_w", "beta_1_w", "beta_2_w", "beta_3_w")
fixed_draws <- eiv_model_fit$draws(fixed_parameters, format = "df") %>%
  pivot_longer(cols = all_of(fixed_parameters), names_to = "variable", values_to = "value")
fixed_fit <- eiv_model_fit$summary(fixed_parameters)

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
random_parameters <- c("Var_alpha_1_z1", "Var_alpha_1_z2", "Rho_slope_z1_slope_z2")
random_draws <- eiv_model_fit$draws(random_parameters, format = "df") %>%
  pivot_longer(cols = all_of(random_parameters), names_to = "variable", values_to = "value")
random_fit <- eiv_model_fit$summary(random_parameters)

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
