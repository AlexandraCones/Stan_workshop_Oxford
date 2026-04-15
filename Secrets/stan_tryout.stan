// ~~~ z1 = beta_0 +  alpha_0[ID] + (beta_1 + alpha_1[ID]) .* x + e ~~~ //
// ~~~ z2 = beta_0 +  alpha_0[ID] + (beta_1 + alpha_1[ID]) .* x + e ~~~ //
// ~~~ w = beta_0_w + beta_1_w*alpha_1_z1[ID] + beta_2_w*alpha_1_z2[ID] + beta_3_w*alpha_1_z1[ID]*alpha_1_z2[ID] + e_w ~~~ //

data {
  int<lower=1> N_z; // number of trait response values 
  int<lower=1> N_w; // number of fitness response values 
  int<lower=1> N_ID; // number of individuals 
  array[N_z] int<lower=1> ID_z; // array of individual ids for trait
  array[N_w] int<lower=1> ID_w; // array of individual ids for fitness
  vector[N_z] x; // environment values 
  vector[N_z] z1; // trait response values 
  vector[N_z] z2; // trait response values 
  vector[N_w] w; // fitness response values 
}

parameters {
  real beta_0_z1; // population intercept trait
  real beta_1_z1; // population slope trait
  real<lower=0> sigma_0_z1; // residual error trait
  
  real beta_0_z2; // population intercept trait
  real beta_1_z2; // population slope trait
  real<lower=0> sigma_0_z2; // residual error trait
  
  real beta_0_w; // population intercept fitness
  real beta_1_w; // selection on intercepts
  real beta_2_w; // selection on slopes
  real beta_3_w; // selection on correlation of intercepts and slopes
  real<lower=0> sigma_0_w; // residual error fitness
  
  vector<lower=0>[4] sd_alpha;       // SDs for intercept, slope
  cholesky_factor_corr[4] R_chol;    // Cholesky of 2x2 correlation matrix
  matrix[N_ID, 4] raw_dev_alpha;     // raw (non-centered) deviations
}

transformed parameters {
  matrix[N_ID, 4] alpha = raw_dev_alpha * diag_pre_multiply(sd_alpha, R_chol)'; // BLUPS
}

model {
  vector[N_z] mu_z1 = beta_0_z1 + col(alpha,1)[ID_z] + (beta_1_z1 + col(alpha,2)[ID_z]) .* x; // mean model trait
  vector[N_z] mu_z2 = beta_0_z2 + col(alpha,3)[ID_z] + (beta_1_z2 + col(alpha,4)[ID_z]) .* x; // mean model trait
  vector[N_w] mu_w = beta_0_w
                 + beta_1_w * col(alpha,2)[ID_w]
                 + beta_2_w * col(alpha,4)[ID_w]
                 + beta_3_w * col(alpha,2)[ID_w] .* col(alpha,4)[ID_w]; // mean model fitness
  
  z1 ~ normal(mu_z1, sigma_0_z1);
  z2 ~ normal(mu_z2, sigma_0_z2);
  w ~ normal(mu_w, sigma_0_w);
  
  beta_0_z1 ~ normal(0,1); // population intercept prior
  beta_1_z1 ~ normal(0,1); // population slope prior
  sigma_0_z1 ~ normal(0,1); // residual error prior
  
  beta_0_z2 ~ normal(0,1); // population intercept prior
  beta_1_z2 ~ normal(0,1); // population slope prior
  sigma_0_z2 ~ normal(0,1); // residual error prior
  
  beta_0_w ~ normal(0,1); // population intercept prior fitness
  beta_1_w ~ normal(0,1); // selection on intercepts prior
  beta_2_w ~ normal(0,1); // selection on slopes prior
  beta_3_w ~ normal(0,1); // selection on correlation of intercepts and slopes prior
  sigma_0_w ~ normal(0,1); // residual error prior
  
  sd_alpha ~ exponential(2); // variance prior
  R_chol   ~ lkj_corr_cholesky(2); // correlation prior
  to_vector(raw_dev_alpha) ~ std_normal(); // deviations prior
}

generated quantities {
  matrix[4,4] R = multiply_lower_tri_self_transpose(R_chol);

  // Variances
  real Var_alpha_0_z1 = square(sd_alpha[1]);
  real Var_alpha_1_z1 = square(sd_alpha[2]);
  real Var_alpha_0_z2 = square(sd_alpha[3]);
  real Var_alpha_1_z2 = square(sd_alpha[4]);

  // Correlations 
  real Rho_int_z1_slope_z1 = R[1,2];  // within-trait z1
  real Rho_int_z2_slope_z2 = R[3,4];  // within-trait z2
  real Rho_int_z1_int_z2   = R[1,3];  // cross-trait intercepts
  real Rho_slope_z1_slope_z2 = R[2,4]; // cross-trait slopes

  // Covariances 
  real Cov_alpha_0_z1_alpha_1_z1 = R[1,2] * sd_alpha[1] * sd_alpha[2];
  real Cov_alpha_0_z2_alpha_1_z2 = R[3,4] * sd_alpha[3] * sd_alpha[4];
}



