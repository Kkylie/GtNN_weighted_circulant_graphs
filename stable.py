import torch
import torch.nn as nn
import sympy as sp
import numpy as np

# compute expansion parameters
def compute_constrain_param(model):
  device = next(model.parameters()).device
  M = model.monomial_word_support
  q_j_alpha_tensor = torch.Tensor(M.num_variables, M.num_monomial_words())
  for ind_word, word in enumerate(M.monomial_words):
    for ind_variable, variable in enumerate(M.variables):
      di = sp.diff(word,variable)
      lis = [(all_variable,1) for all_variable in M.variables]
      q_j_alpha_tensor[ind_variable,ind_word] = torch.FloatTensor([di.subs(lis).evalf()])
  q_j_alpha_tensor = q_j_alpha_tensor.to(device)

  C_max_sum_all_layer = []
  C_j_max_sum_all_layer = []
  for param in model.parameters():
    param_result = param
    C_j_h_tensor = torch.tensordot(q_j_alpha_tensor,torch.abs(param_result), dims = ([1],[2]))
    C_h_tensor = torch.sum(torch.abs(param_result), dim = 2)
    C_max_sum = torch.max(torch.sum(C_h_tensor,dim = 1))
    C_j_max_sum = torch.max(torch.sum(C_j_h_tensor,dim = 2),dim = 1)
    C_max_sum_all_layer.append(C_max_sum)
    C_j_max_sum_all_layer.append(C_j_max_sum[0])
  return C_max_sum_all_layer, C_j_max_sum_all_layer

# compute stablilty constraint penalty
def compute_penalty(model, Upr_Cj_vec, Upr_C = 1, constrain_C_Flag = True, constrain_Cj_Flag = True):
  device = next(model.parameters()).device
  loss_penalty = 0
  [C_max_sum_all_layer, C_j_max_sum_all_layer] = compute_constrain_param(model)

  if Upr_C == 1:
    Upr_C = []
    for ind_layer, C_max_sum_each_layer in enumerate(C_max_sum_all_layer):
      Upr_C.append(1)
    Upr_C= Upr_C.to(device)

  f = nn.ReLU()
  if constrain_C_Flag == True:
    for ind_layer, C_max_sum_each_layer in enumerate(C_max_sum_all_layer):
      loss_penalty = loss_penalty + f(C_max_sum_each_layer - Upr_C[ind_layer])
  if constrain_Cj_Flag == True:
    for ind_layer, C_j_max_sum_each_layer in enumerate(C_j_max_sum_all_layer):
      Upr_Cj_vec_each = Upr_Cj_vec[ind_layer]
      for ind_variable, C_j_each in enumerate(C_j_max_sum_each_layer):
        loss_penalty = loss_penalty + f(C_j_each - Upr_Cj_vec_each[ind_variable])
  return loss_penalty

# compute stability metric with respect to input signal perturbation
def compute_h_poly_matrix_norm(model,num_vertices):
  device = next(model.parameters()).device
  X_id = torch.eye(num_vertices).reshape(num_vertices,1,num_vertices)
  X_id = X_id.to(device)
  model.eval()
  with torch.no_grad():
    h_poly_matrix = model.forward(X_id)
    h_poly_matrix = h_poly_matrix.reshape(num_vertices,num_vertices)
    h_poly_matrix_norm = torch.linalg.matrix_norm(h_poly_matrix,ord = 2)
  return h_poly_matrix_norm

# create perturbed graph tuple for computing stability metric
def create_perturbation_Z(operator_tuple):
  trainStabilityEpsilon = 0.5
  actual_peturbation_size = np.zeros(2)
  ts_perturb = []
  for ind_g, g in enumerate(operator_tuple):
    actual_peturbation_size[ind_g] = trainStabilityEpsilon
    if trainStabilityEpsilon > 0:
      S = g
      E = torch.rand(size=S.shape)
      E = torch.triu(E) + torch.triu(E,diagonal = 1).t()
      E = trainStabilityEpsilon * E / torch.linalg.matrix_norm(E,ord = 2)
      S_hat= S + E
    else:
      S_hat = g
    if torch.linalg.matrix_norm(S_hat,ord = 2) > 1:
      S_hat = S_hat / torch.linalg.matrix_norm(S_hat,ord = 2)
      E_actual = S_hat - S

      trainStabilityEpsilon_actual = torch.linalg.matrix_norm(E_actual,ord = 2)
      actual_peturbation_size[ind_g] = trainStabilityEpsilon_actual
    ts_perturb.append(S_hat)
  ts_perturb_tuple = (ts_perturb[0],ts_perturb[1])

  return ts_perturb_tuple, actual_peturbation_size

# compute stability metric with respect to graph perturbation
def compute_hw_minus_hz_poly_matrix_norm(model, num_vertices, operator_tuple, ts_perturb_tuple):
  device = next(model.parameters()).device
  X_id = torch.eye(num_vertices).reshape(num_vertices,1,num_vertices)
  X_id = X_id.to(device)
  M = model.monomial_word_support
  model.eval()
  with torch.no_grad():
    M.evaluate_at_operator_tuple(operator_tuple = ts_perturb_tuple)
    model.change_monomial_word_support(M)
    h_Z = model.forward(X_id)
    h_Z = h_Z.reshape(num_vertices,num_vertices)

    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    model.change_monomial_word_support(M)
    h_W = model.forward(X_id)
    h_W = h_W.reshape(num_vertices,num_vertices)

    hw_minus_hz_poly_matrix_norm = torch.linalg.matrix_norm((h_W - h_Z),ord = 2)

  return hw_minus_hz_poly_matrix_norm