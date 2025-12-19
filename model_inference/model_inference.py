from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

"""
This code is to run the trained model on the test and calibration
to get the output resylts for further analysis
"""

test_set = torch.load("data/test_set2.pt",weights_only=False)
calibration_set = torch.load("data/calibration_set2.pt",weights_only=False)

class MyHeteroGNNShared(nn.Module):
    # GNN model for learning uncertainty
    def __init__(self, hidden_dim, answer_embed_dim, num_answers, num_layers=6):
        super().__init__()
        self.num_layers = num_layers

        # learnable embedding for answer nodes
        self.answer_emb = nn.Embedding(num_answers, answer_embed_dim)

        # Input projections
        self.step_proj = nn.Linear(384, hidden_dim)
        self.answer_proj = nn.Linear(answer_embed_dim, hidden_dim)

        # First set of hetero convolutional blocks
        self.conv1 = HeteroConv({
            ('step', 'implies', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'semantic', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('answer', 'equivalent', 'answer'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'contributes', 'answer'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        # second set of hetero convolutional blocks
        self.conv2 = HeteroConv({
            ('step', 'implies', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'semantic', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('answer', 'equivalent', 'answer'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'contributes', 'answer'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        # Feed forward for final answer prediction
        self.answer_predictor = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, data: HeteroData):
        # Initial projection to hidden_dim
        x_dict = {
            'step': self.step_proj(data['step'].x),           
            'answer': self.answer_proj(self.answer_emb.weight)
        }

        # Reuse conv1 and conv2 alternately
        for i in range(self.num_layers):
            if i % 2 == 0:
                x_dict = self.conv1(x_dict, data.edge_index_dict)
            else:
                x_dict = self.conv2(x_dict, data.edge_index_dict)
            x_dict = {k: self.act(x) for k, x in x_dict.items()}

        # feed forward for scalar predcition
        answer_scores = F.sigmoid(self.answer_predictor(x_dict['answer']))
        return answer_scores.squeeze(-1)


# model hyper parameters
hidden_dim = 32
answer_embed_dim = 8
num_layers = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
num_answers = 10
model = MyHeteroGNNShared(hidden_dim, answer_embed_dim, num_answers, num_layers)
model = model.to(device)
print("loading model")
model.load_state_dict(torch.load("models/GNN_model4.pt", map_location='cpu'))

criterion = nn.BCELoss()
model.eval()

all_preds = []
all_golds = []
total_loss = 0.0

print("running test set")
with torch.no_grad():
    # go through test set and save true values with model output
    for data_graph, gold_answers in test_set:
        data_graph = data_graph.to(device)
        gold_answers = gold_answers.to(device).float()

        pred_answers = model(data_graph) 
        loss = criterion(pred_answers, gold_answers)

        total_loss += loss.item()

        # Store flattened
        all_preds.append(pred_answers.detach().cpu())
        all_golds.append(gold_answers.detach().cpu())

avg_loss = total_loss / len(test_set)
print("test_loss:", avg_loss)


pred_vec = torch.cat(all_preds, dim=0)
gold_vec = torch.cat(all_golds, dim=0)

results = torch.stack([pred_vec, gold_vec], dim=1)

print(results.shape) 
torch.save(results, "test_predictions.pt")

print("running calibration set")

## calibreation set

all_preds = []
all_golds = []
total_loss = 0.0

with torch.no_grad():
    for data_graph, gold_answers in calibration_set:
        # go through calibration set and save model output with true label
        data_graph = data_graph.to(device)
        gold_answers = gold_answers.to(device).float()

        pred_answers = model(data_graph)  
        loss = criterion(pred_answers, gold_answers)

        total_loss += loss.item()

        # Store flattened
        all_preds.append(pred_answers.detach().cpu())
        all_golds.append(gold_answers.detach().cpu())

avg_loss = total_loss / len(calibration_set)
print("calibration_loss:", avg_loss)


pred_vec = torch.cat(all_preds, dim=0)
gold_vec = torch.cat(all_golds, dim=0)

results = torch.stack([pred_vec, gold_vec], dim=1)

print(results.shape) 
torch.save(results, "calibration_predictions.pt")

