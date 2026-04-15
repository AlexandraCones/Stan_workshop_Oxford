// z = beta_0 + alpha_0[ID] + e,  e ~ normal(0, exp(log_sigma_0 + alpha_sigma[ID]))
// w = beta_0_w + beta_1_w*alpha_sigma[ID] + e_w

data {
  int<lower=1> N_z;
  int<lower=1> N_w;
  int<lower=1> N_ID;
  array[N_z] int<lower=1> ID_z;
  array[N_w] int<lower=1> ID_w;
  vector[N_z] z;
  vector[N_w] w;
}

parameters {
  real beta_0;               // population intercept trait
  real log_sigma_0;          // population log residual SD
  
  real beta_0_w;             // population intercept fitness
  real beta_1_w;             // selection on predictability
  real beta_2_w;  // quadratic (stabilizing) selection on predictability
  real<lower=0> sigma_0_w;   // residual error fitness

  vector<lower=0>[2] sd_alpha;     // SDs: intercept, log-residual
  cholesky_factor_corr[2] R_chol;  // Cholesky of 2x2 correlation matrix
  matrix[N_ID, 2] raw_dev_alpha;   // raw (non-centered) deviations
}

transformed parameters {
  matrix[N_ID, 2] alpha = raw_dev_alpha * diag_pre_multiply(sd_alpha, R_chol)';
}

model {
  vector[N_z] mu_z = beta_0 + col(alpha,1)[ID_z];
  vector[N_z] sigma_z = exp(log_sigma_0 + col(alpha,2)[ID_z]);
  
  vector[N_w] mu_w = beta_0_w + beta_1_w * col(alpha,2)[ID_w] + beta_2_w * square(col(alpha,2)[ID_w]);

  z ~ normal(mu_z, sigma_z);
  w ~ normal(mu_w, sigma_0_w);

  beta_0  ~ normal(0,1);
  log_sigma_0  ~ normal(0,1);

  beta_0_w  ~ normal(0,1);
  beta_1_w  ~ normal(0,1);
  beta_2_w ~ normal(0,1);
  sigma_0_w ~ normal(0,1); // half-normal prior

  sd_alpha ~ exponential(2);
  R_chol   ~ lkj_corr_cholesky(2);
  to_vector(raw_dev_alpha) ~ std_normal();
}
generated quantities {
  matrix[2,2] R = multiply_lower_tri_self_transpose(R_chol);

  real Var_alpha_0        = square(sd_alpha[1]);  // variance in intercepts
  real Var_alpha_sigma    = square(sd_alpha[2]);  // variance in predictability
  real Rho_alpha_0_sigma  = R[1,2];               // correlation between intercept and predictability
  real Cov_alpha_0_sigma  = R[1,2] * sd_alpha[1] * sd_alpha[2];
}