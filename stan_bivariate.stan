// ~~~ z = beta_0 +  alpha_0[ID] + (beta_1 + alpha_1[ID]) .* x + e ~~~ //
// ~~~ w = beta_0_w +  alpha_0_w[ID]  + e_w ~~~ //

data {
  int<lower=1> N_z; // number of trait response values 
  int<lower=1> N_w; // number of fitness response values 
  int<lower=1> N_ID; // number of individuals 
  array[N_z] int<lower=1> ID_z; // array of individual ids for trait
  array[N_w] int<lower=1> ID_w; // array of individual ids for fitness
  vector[N_z] x; // environment values 
  vector[N_z] z; // trait response values 
  vector[N_w] w; // fitness response values 
}

parameters {
  real beta_0; // population intercept trait
  real beta_1; // population slope trait
  real<lower=0> sigma_0; // residual error trait
  
  real beta_0_w; // population intercept fitness
  real<lower=0> sigma_0_w; // residual error fitness
  
  vector<lower=0>[3] sd_alpha;       // SDs for intercept, slope, fitness intercept
  cholesky_factor_corr[3] R_chol;    // Cholesky of 3x3 correlation matrix
  matrix[N_ID, 3] raw_dev_alpha;     // raw (non-centered) deviations
}

transformed parameters {
  matrix[N_ID, 3] alpha = raw_dev_alpha * diag_pre_multiply(sd_alpha, R_chol)'; // BLUPS
}

model {
  vector[N_z] mu = beta_0 + col(alpha,1)[ID_z] + (beta_1 + col(alpha,2)[ID_z]) .* x; // mean model trait
  vector[N_w] mu_w = beta_0_w + col(alpha,3)[ID_w]; // mean model fitness
  
  z ~ normal(mu, sigma_0);
  w ~ normal(mu_w, sigma_0_w);
  
  beta_0 ~ normal(0,1); // population intercept prior
  beta_1 ~ normal(0,1); // population slope prior
  sigma_0 ~ normal(0,1); // residual error prior
  
  beta_0_w ~ normal(0,1); // population intercept prior fitness
  sigma_0_w ~ normal(0,1); // residual error prior
  
  sd_alpha ~ exponential(2); // variance prior
  R_chol   ~ lkj_corr_cholesky(2); // correlation prior
  to_vector(raw_dev_alpha) ~ std_normal(); // deviations prior
}

generated quantities {
  matrix[3,3] R = multiply_lower_tri_self_transpose(R_chol); // extract correlation matrix
  
  real Rho_alpha_0_alpha_0_w = R[1,3]; // corr trait intercepts and fitness
  real Rho_alpha_1_alpha_0_w = R[2,3]; // corr trait slopes and fitness
  real Rho_alpha_0_alpha_1 = R[1,2]; // corr trait intercepts and slopes

  real Cov_alpha_0_alpha_0_w = R[1,3] * sd_alpha[1] * sd_alpha[3]; // covariance trait intercepts and fitness
  real Cov_alpha_1_alpha_0_w = R[2,3] * sd_alpha[2] * sd_alpha[3]; // covariance trait slopes and fitness
  real Cov_alpha_0_alpha_1 = R[1,2] * sd_alpha[1] * sd_alpha[2]; //  covariance trait intercepts and slopes

  real Var_alpha_0 = square(sd_alpha[1]); // variance trait intercepts
  real Var_alpha_1 = square(sd_alpha[2]); // variance trait slopes
  real Var_alpha_0_w = square(sd_alpha[3]); // variance fitness intercepts
}



