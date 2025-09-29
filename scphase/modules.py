# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def initialize_weights(module):
    for m in module.modules():
        if not hasattr(m, 'weight') or not m.weight.requires_grad:
            continue
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            if m.weight.requires_grad:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None and m.bias.requires_grad:
                nn.init.constant_(m.bias, 0)

class InstanceDropout(nn.Module):
    def __init__(self, dropout_rate=0.15):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, instances):
        if self.training and instances.size(0) > 1:
            num_instances = instances.size(0)
            num_keep = max(1, int(num_instances * (1 - self.dropout_rate)))
            indices = torch.randperm(num_instances)[:num_keep]
            return instances[indices]
        return instances

class AdaptiveMILAggregation(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        
        self.attention = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.stat_aggregator = nn.Linear(input_dim * 3, input_dim)
        
    def forward(self, instances):
        attention_weights = F.softmax(self.attention(instances), dim=0)
        gated_instances = instances * self.gate(instances)
        attended_features = torch.sum(attention_weights * gated_instances, dim=0)
        stat_features = torch.cat([torch.mean(instances, dim=0), torch.max(instances, dim=0)[0], torch.std(instances, dim=0)], dim=0)
        stat_features = self.stat_aggregator(stat_features)
        return attended_features + stat_features, attention_weights

class MoEMILAggregation(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_experts=4, dropout_rate=0.15):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1)) for _ in range(num_experts)])
        self.gate_network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, num_experts))
        
    def forward(self, instances):
        enhanced_instances = instances  
        gate_scores = self.gate_network(enhanced_instances)
        gate_weights = F.softmax(gate_scores, dim=-1)
        expert_scores = torch.cat([expert(instances) for expert in self.experts], dim=-1)
        combined_scores = torch.sum(gate_weights * expert_scores, dim=-1, keepdim=True)
        attention_weights = F.softmax(combined_scores, dim=0)
        bag_features = torch.sum(attention_weights * instances, dim=0)
        return bag_features, attention_weights

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

class DomainClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_domains=None):
        super().__init__() 
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    def forward(self, x):
        return self.domain_classifier(x)

class MILLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, bag_predictions, bag_labels, instance_features=None):
        return self.ce_loss(bag_predictions, bag_labels)