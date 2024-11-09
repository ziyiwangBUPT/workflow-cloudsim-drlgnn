# """
# @startuml

# start


# partition Input {
# :Inputs: action_state, hidden_state, machine_action_mask, machine_task_time;

# :Normalize features;
# note right: Scales input values machine_task_time and action_state\nby dividing by et_normalize_coef
# }

# partition "Full Connected Layer" {
# :Concatenate features for transformation;
# note right: Concatenates machine_task_time and action_state \n into a feature vector.

# :Apply Linear Transformation
# Projects the feature vector to increase
# representation capacity.;
# note right: Transforms concatenated features \n using a linear layer into ft_out.
# }

# partition "Batch Normalization" {
# :Apply Batch Normalization
# Reduces internal covariate shift, stabilizing
#  training and allowing faster convergence.;
# note right: Normalizes ft_out \n to have mean 0 and variance 1.
# }

# partition "MLP Decoder" {
# :Compute Mean of Normalized Values
# Creates a global summary of machine features
# to use as context in decision making.;
# note right: Averages features across n_machines into pooled_features, \n creating a pooled representation.

# :Concatenate Transformed Features by
# Repeating Pool and Hidden State;
# note right: Expands pooled_features and hidden_state to match dimensions \n of action_node and concatenates action_node, repeated pooled_features, \n and repeated hidden_state

# :Compute Machine Scores that reflect the
# desirability of each machine choice.;
# note right: Passes concatenated features through MLPActor \n for scoring each machine action.

# :Apply Mask to Machine Scores
# Prevents selection of invalid actions.;
# note right: Sets invalid machine actions to -inf \n based on machine_action_mask.

# :Apply Softmax for Action Probability;
# note right: Converts scores to probability distribution \n over machine actions.
# }

# stop

# @enduml
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from icecream import ic

# from gym_simulator.algorithms.graph.mlp import MLPActor


# class MachineActor(nn.Module):
#     """
#     Machine Actor model for machine scheduling tasks in reinforcement learning.
#     """

#     def __init__(
#         self,
#         n_jobs: int,
#         n_machines: int,
#         hidden_dim: int,
#         device: torch.device = torch.device("cpu"),
#         et_normalize_coef=1000,
#     ):
#         super().__init__()
#         self.n_jobs = n_jobs
#         self.n_machines = n_machines
#         self.hidden_dim = hidden_dim
#         self.device = device
#         self.et_normalize_coef = et_normalize_coef

#         # Input transformations and batch normalization
#         self.batch_norm = nn.BatchNorm1d(hidden_dim).to(device)
#         self.feature_transformer = nn.Linear(2, hidden_dim, bias=False).to(device)

#         # Actor network
#         self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)

#     def forward(
#         self,
#         machine_incompatibility_mask: torch.Tensor,
#         operation_proc_time: torch.Tensor,
#         machine_completion_time: torch.Tensor,
#         machine_pooling_vector: torch.Tensor,
#     ):
#         """
#         Forward pass for MachineActor.

#         @param machine_completion_time: T_t(M_k) Completion time for each machine (batch_size, n_machines)
#         @param operation_proc_time: p_ijk Processing time for the selected operation (batch_size, n_machines)
#         @param machine_incompatibility_mask: Mask for incompatible machines for the selected operation (batch_size, n_machines)
#         @param machine_pooling_vector: Hidden state of the machine actor (batch_size, hidden_size) (????)

#         @return machine_action_probabilities: Probability distribution over machine actions (batch_size, n_machines)
#         @return global_graph_embedding: Pooled features of machine actions (batch_size, hidden_size)
#         """

#         # Normalize inputs and contactenate features
#         machine_completion_time = machine_completion_time / self.et_normalize_coef
#         operation_proc_time = operation_proc_time / self.et_normalize_coef
#         features = torch.cat([machine_completion_time.unsqueeze(-1), operation_proc_time.unsqueeze(-1)], -1)

#         # h^t_k: Apply Linear Transformation and Batch Normalization
#         ft_out: torch.Tensor = self.feature_transformer(features)
#         bn_out: torch.Tensor = self.batch_norm(ft_out.reshape(-1, self.hidden_dim))
#         machine_node_embedding = bn_out.reshape(-1, self.n_machines, self.hidden_dim)

#         # h^t_G: Compute mean of the final node embeddings
#         global_graph_embedding = machine_node_embedding.mean(dim=1)

#         # Repeat pooled features and concatenate for actor
#         global_graph_embedding_rep = global_graph_embedding.unsqueeze(1).expand_as(machine_node_embedding)
#         machine_pooling_vector_rep = machine_pooling_vector.unsqueeze(1).expand_as(machine_node_embedding)
#         concat_feats = torch.cat(
#             # h^t_k               || h^t_G                    || u_t
#             (machine_node_embedding, global_graph_embedding_rep, machine_pooling_vector_rep),
#             dim=-1,
#         )
#         ic(concat_feats.shape)

#         # c^m_t,k: Calculate scores and apply mask
#         machine_action_scores: torch.Tensor = self.actor(concat_feats) * 10
#         machine_action_scores = machine_action_scores.squeeze(-1)
#         machine_action_scores = machine_action_scores.masked_fill(
#             machine_incompatibility_mask.squeeze(1).bool(), float("-inf")
#         )
#         ic(machine_action_scores.shape)

#         # p_j(a^m_t): Softmax for probability distribution
#         machine_action_probabilities = F.softmax(machine_action_scores, dim=1)
#         ic(machine_action_probabilities.shape)
#         ic(machine_action_probabilities[0])

#         return machine_action_probabilities, global_graph_embedding


# if __name__ == "__main__":
#     torch.manual_seed(42)
#     machine_actor = MachineActor(n_jobs=10, n_machines=5, hidden_dim=512)
#     result = machine_actor(
#         machine_pooling_vector=torch.rand(64, 512),
#         machine_incompatibility_mask=torch.zeros(64, 5),
#         operation_proc_time=torch.rand(64, 5),
#         machine_completion_time=torch.rand(64, 5),
#     )
#     assert result[0].shape == (64, 5)
#     assert result[1].shape == (64, 512)
