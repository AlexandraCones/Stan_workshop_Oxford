// ~~~ z = beta_0 +  alpha_0[ID] + (beta_1 + alpha_1[ID]) .* x + e ~~~ //

data {
  int<lower=1> N_z; // number of response values 
  int<lower=1> N_ID; // number of individuals 
  array[N_z] int<lower=1> ID; // array of individual ids
  vector[N_z] x; // environment values 
  vector[N_z] z; // response values 
}

parameters {
  real beta_0; // population intercept 
  real beta_1; // population slope 
  vector<lower=0>[2] sd_alpha; // intercept sd and slope standard deviations
  matrix[N_ID,2] raw_dev_alpha; // raw individual-level intercept and slope deviations
  cholesky_factor_corr[2] R_chol; // cholesky decomposition of 2x2 intercept-slope correlation matrix
  real<lower=0> sigma_0; // residual error
}

transformed parameters {
  matrix[N_ID,2] alpha = raw_dev_alpha * diag_pre_multiply(sd_alpha, R_chol)' ; // BLUPs 
}

model {
  vector[N_z] mu = beta_0 + col(alpha,1)[ID] + (beta_1 + col(alpha,2)[ID]) .* x; // mean model
  vector[N_z] sigma = rep_vector(sigma_0, N_z); // error model
  z ~ normal(mu,sigma); // the likelihood

  beta_0 ~ normal(0,1); // population intercept prior
  beta_1 ~ normal(0,1); // population slope prior
  sd_alpha ~ exponential(2); // intercept and slope standard deviations priors
  R_chol ~ lkj_corr_cholesky(2); // random effect correlation prior
  to_vector(raw_dev_alpha) ~ std_normal(); // individual intercept and slope deivations priors
  sigma_0 ~ normal(0,1); // residual error prior
}

generated quantities {
  matrix[2,2] R = multiply_lower_tri_self_transpose(R_chol); // extract correlation matrix
  real Rho_alpha_01 = R[1,2]; // extract correlation
  real Cov_alpha_01 = Cor_alpha_01 * sd_alpha[1] * sd_alpha[2]; // calculate covariance
  real Var_alpha_0 = square(sd_alpha[1]); // calculate variance
  real Var_alpha_1 = square(sd_alpha[2]); // calculate variance
}

