import json, re
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

"""
This code is for training the GNN classifier model
"""

def build_hetero_graph(questions, embedder):
    """
    build the heterogenous graph structure
    """

    data = HeteroData()

    step_texts = []
    step_chain_id = []  
    step_idx_map = [] 

    answer_texts = []
    answer_chain_id = []

    # build graph nodes
    for c, resp in enumerate(questions["responses"]):
        steps = resp["steps"]
        final = resp["pred_answer"]

        for s_idx, step in enumerate(steps):
            step_texts.append(step)
            step_chain_id.append(c)
            step_idx_map.append((c, s_idx))

        answer_texts.append(final)
        answer_chain_id.append(c)

    # use sentence embeddings
    step_emb = torch.tensor(embedder.encode(step_texts), dtype=torch.float)
    data["step"].x = step_emb
    data["answer"].x = torch.tensor(embedder.encode(answer_texts), dtype=torch.float)

    # store chain flag
    data["step"].chain_id = torch.tensor(step_chain_id, dtype=torch.long)
    data["answer"].chain_id = torch.tensor(answer_chain_id, dtype=torch.long)

    return data, step_idx_map, answer_texts


def add_implication_edges(data, questions, step_idx_map):
    """
    Add step implication edges in the hetero graph
    """
    edges = []

    for c, resp in enumerate(questions["responses"]):
        steps = resp["steps"]
        for s in range(len(steps) - 1):
            src = step_idx_map.index((c, s))
            dst = step_idx_map.index((c, s+1))
            edges.append([src, dst])

    if edges:
        data["step", "implies", "step"].edge_index = torch.tensor(edges).t().contiguous()
    else:
        data["step", "implies", "step"].edge_index = torch.empty((2,0), dtype=torch.long)

    return data



def add_semantic_edges(data, sim_threshold=0.90):
    """
    Add semantic equivelance edges in the hetero graph
    """
    X = data["step"].x
    sims = F.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=-1)

    edges = []
    scores = []

    N = sims.shape[0]
    for i in range(N):
        for j in range(i+1, N):  # avoid duplicates & self
            if sims[i, j] >= sim_threshold:
                edges.append([i, j])
                edges.append([j, i])  # make symmetric
                scores.append(sims[i, j].item())
                scores.append(sims[i, j].item())

    if edges:
        edges = torch.tensor(edges).t().contiguous()
        scores = torch.tensor(scores).unsqueeze(1)
    else:
        edges = torch.empty((2,0), dtype=torch.long)
        scores = torch.empty((0,1), dtype=torch.float)

    data["step", "semantic", "step"].edge_index = edges
    data["step", "semantic", "step"].edge_attr = scores

    return data


def add_answer_equivalence_edges(data, answer_texts):
    """
    Add same answer edges in the hetero graph
    """
    edges = []

    N = len(answer_texts)
    for i in range(N):
        for j in range(i+1, N):
            if answer_texts[i].strip() == answer_texts[j].strip():
                edges.append([i, j])
                edges.append([j, i])

    if edges:
        edges = torch.tensor(edges).t().contiguous()
    else:
        edges = torch.empty((2,0), dtype=torch.long)

    data["answer", "equivalent", "answer"].edge_index = edges
    return data

def add_step_to_answer_edges(data, step_idx_map):
    """
    Add stpe implies answer edges in the hetero graph
    """
    edges = []
    for s_idx, (chain_id, _) in enumerate(step_idx_map):
        edges.append([s_idx, chain_id])

    if edges:
        data["step", "contributes", "answer"].edge_index = torch.tensor(edges).t().contiguous()
    else:
        data["step", "contributes", "answer"].edge_index = torch.empty((2,0), dtype=torch.long)

    return data


def build_full_graph(questions, embedder, sim_threshold = 0.90):
    """
    create full hetero graph for model training
    """
    data, step_idx_map, answer_texts = build_hetero_graph(questions, embedder)

    data = add_implication_edges(data, questions, step_idx_map)
    data = add_semantic_edges(data, sim_threshold=sim_threshold)
    data = add_answer_equivalence_edges(data, answer_texts)
    data = add_step_to_answer_edges(data, step_idx_map)

    return data



def split_response_into_steps(text):
    """
    function to split a response into chain of thought steps
    """

    # aremove stawrt to make easier
    text = text.replace('[/INST]', '\n\n')

    # split based on \n
    raw_steps = re.split(r'\n+', text)
    steps = [s.strip() for s in raw_steps if s.strip()]

    # skip input prompt
    return steps[2:]



## Load data
data_train = []
with open("data/gsm8k_inference_results_train1.jsonl", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            try:
                data_train.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line)


with open("data/gsm8k_inference_results_train2.jsonl", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            try:
                data_train.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line)


"""
This coded out chunk is to build the graphs and save the graphs
"""
# def add_steps(data):
#     for q in range(len(data)):
#         for i in range(10):
#             response = data[q]['responses'][i]['raw_response']
#             data[q]['responses'][i]['steps'] = split_response_into_steps(response)

#     return data

## Load embedder
# from sentence_transformers import SentenceTransformer
# embedder = SentenceTransformer('models/all-MiniLM-L6-v2')

## Prep data
# data_train = add_steps(data_train)
# get_indeces = range(6000)
# sim_threshold = 0.90
# training_set = []
# for i in get_indeces:
#    t_graph = build_full_graph(data_train[i],embedder,sim_threshold=sim_threshold)
#    gold_answer = torch.tensor([resp['correct'] for resp in data_train[i]['responses']])
#    training_set.append([t_graph,gold_answer])

# print("saved training set")
# torch.save(training_set,"training_set2.pt")
"""
Just load in graphs if they have already been built
"""

training_set = torch.load("training_set2.pt",weights_only=False)
# split training and eval
split = int(0.8 * len(training_set))
eval_set   = training_set[split:]
training_set = training_set[:split]


## Define GNN model
class AnswerNodeEmbedding(nn.Module):
    def __init__(self, num_answers, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(num_answers, embed_dim)

    def forward(self, batch_size=None):
        return self.emb.weight


class MyHeteroGNNShared(nn.Module):
    def __init__(self, hidden_dim, answer_embed_dim, num_answers, num_layers=6):
        super().__init__()
        self.num_layers = num_layers

        self.answer_emb = nn.Embedding(num_answers, answer_embed_dim)

        self.step_proj = nn.Linear(384, hidden_dim)
        self.answer_proj = nn.Linear(answer_embed_dim, hidden_dim)

        self.conv1 = HeteroConv({
            ('step', 'implies', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'semantic', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('answer', 'equivalent', 'answer'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'contributes', 'answer'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('step', 'implies', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'semantic', 'step'): SAGEConv((-1, -1), hidden_dim),
            ('answer', 'equivalent', 'answer'): SAGEConv((-1, -1), hidden_dim),
            ('step', 'contributes', 'answer'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        self.answer_predictor = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, data: HeteroData):
        x_dict = {
            'step': self.step_proj(data['step'].x),        
            'answer': self.answer_proj(self.answer_emb.weight) 
        }

        for i in range(self.num_layers):
            if i % 2 == 0:
                x_dict = self.conv1(x_dict, data.edge_index_dict)
            else:
                x_dict = self.conv2(x_dict, data.edge_index_dict)
            x_dict = {k: self.act(x) for k, x in x_dict.items()}

        answer_scores = F.sigmoid(self.answer_predictor(x_dict['answer']))
        return answer_scores.squeeze(-1)


"""
Loop to train GNN binary classifier model
"""
# Hyperparameters
hidden_dim = 32
answer_embed_dim = 8
num_layers = 6
lr = 1e-2
num_epochs = 500
eval_freq = 10
best_eval_loss = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_answers = 10
model = MyHeteroGNNShared(hidden_dim, answer_embed_dim, num_answers, num_layers)
model = model.to(device)

# Optimizer and loss
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.BCELoss()


print("Starting Training")
print("training set length: ", len(training_set))
print("epochs: ",num_epochs)
print("learning rate: ",lr)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    # update parameters on training set, one  at a time since batching was being wierd with the graphs
    for example in training_set:
        data_graph, gold_answers = example  # unpack
        data_graph = data_graph.to(device)
        gold_answers = gold_answers.to(device).float()  # ensure float for regression

        optimizer.zero_grad()
        pred_answers = model(data_graph)  # shape [num_answers]
        loss = criterion(pred_answers, gold_answers)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(training_set)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")


    # Evaluate on the eval set and only save when this loss is minimized
    if epoch % eval_freq ==0:
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for example in eval_set:
                data_graph, gold_answers = example
                data_graph = data_graph.to(device)
                gold_answers = gold_answers.to(device).float() 

                pred_answers = model(data_graph)
                loss = criterion(pred_answers, gold_answers)

                total_loss += loss.item()


        avg_loss = total_loss / len(eval_set)
        print(f"Eval Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        if avg_loss <= best_eval_loss:
            best_eval_loss = avg_loss
            torch.save(model.state_dict(), "models/GNN_model4.pt")

