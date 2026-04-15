// ~~~ z = beta_0 +  alpha_0[ID] + (beta_1 + alpha_1[ID]) .* x + e ~~~ //
// ~~~ w = beta_0_w + alpha_2[ID] + beta_1_w*alpha_0_z[ID] + beta_2_w*alpha_1_z[ID] + beta_3_w*alpha_0_z[ID]*alpha_1_z[ID] + e_w ~~~ //
data {
  int<lower=1> N_z;
  int<lower=1> N_w;
  int<lower=1> N_ID;
  array[N_z] int<lower=1> ID_z;
  array[N_w] int<lower=1> ID_w;
  vector[N_z] x;
  vector[N_z] z;
  vector[N_w] w;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma_0;
  
  real beta_0_w;
  real beta_1_w;
  real beta_2_w;
  real beta_3_w;
  real<lower=0> sigma_0_w;
  
  vector<lower=0>[3] sd_alpha;       // SDs for: trait intercept, trait slope, fitness intercept
  cholesky_factor_corr[3] R_chol;    // Cholesky of 3x3 correlation matrix
  matrix[N_ID, 3] raw_dev_alpha;     // raw (non-centered) deviations
}
transformed parameters {
  matrix[N_ID, 3] alpha = raw_dev_alpha * diag_pre_multiply(sd_alpha, R_chol)'; // BLUPs
  // col(alpha,1) = trait intercepts (alpha_0)
  // col(alpha,2) = trait slopes     (alpha_1)
  // col(alpha,3) = fitness intercepts (alpha_2)
}
model {
  vector[N_z] mu = beta_0 + col(alpha,1)[ID_z] + (beta_1 + col(alpha,2)[ID_z]) .* x;
  vector[N_w] mu_w = beta_0_w
                 + col(alpha,3)[ID_w]                                          // random fitness intercept
                 + beta_1_w * col(alpha,1)[ID_w]
                 + beta_2_w * col(alpha,2)[ID_w]
                 + beta_3_w * col(alpha,1)[ID_w] .* col(alpha,2)[ID_w];
  
  z ~ normal(mu, sigma_0);
  w ~ normal(mu_w, sigma_0_w);
  
  beta_0   ~ normal(0,1);
  beta_1   ~ normal(0,1);
  sigma_0  ~ normal(0,1);
  
  beta_0_w ~ normal(0,1);
  beta_1_w ~ normal(0,1);
  beta_2_w ~ normal(0,1);
  beta_3_w ~ normal(0,1);
  sigma_0_w ~ normal(0,1);
  
  sd_alpha ~ exponential(2);
  R_chol   ~ lkj_corr_cholesky(2);
  to_vector(raw_dev_alpha) ~ std_normal();
}
generated quantities {
  matrix[3,3] R = multiply_lower_tri_self_transpose(R_chol);
  
  // trait intercept - trait slope
  real Rho_alpha_0_alpha_1 = R[1,2];
  real Cov_alpha_0_alpha_1 = R[1,2] * sd_alpha[1] * sd_alpha[2];
  real Var_alpha_0 = square(sd_alpha[1]);
  real Var_alpha_1 = square(sd_alpha[2]);
  
  // fitness intercept correlations with trait parameters
  real Rho_alpha_2_alpha_0 = R[3,1]; // corr fitness intercept and trait intercept
  real Rho_alpha_2_alpha_1 = R[3,2]; // corr fitness intercept and trait slope
  real Cov_alpha_2_alpha_0 = R[3,1] * sd_alpha[3] * sd_alpha[1];
  real Cov_alpha_2_alpha_1 = R[3,2] * sd_alpha[3] * sd_alpha[2];
  real Var_alpha_2 = square(sd_alpha[3]);
}