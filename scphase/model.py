# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import (
    InstanceDropout, AdaptiveMILAggregation, MoEMILAggregation, 
    GradientReversalLayer, DomainClassifier, initialize_weights
)

class SCMIL_AttnMoE(nn.Module):

    def __init__(self, model_params, ablation_params, num_domains, device_encoder, device_model):
        super(SCMIL_AttnMoE, self).__init__()
        
        self.max_instances = model_params['max_instances']
        self.use_moe = ablation_params['use_moe']
        self.attention_type = ablation_params['attention_type']
        self.device_encoder = device_encoder
        self.device_model = device_model
        
        input_dim = model_params['input_dim']
        encoder_dims = model_params['encoder_dims']
        hidden_dim = model_params['hidden_dim']
        encoder_dropout_rates = model_params['instance_encoder_dropout_rates']

        encoder_layers = [
            nn.Linear(input_dim, encoder_dims[0]), 
            nn.ReLU(), 
            nn.Dropout(encoder_dropout_rates[0]),
            
            nn.Linear(encoder_dims[0], encoder_dims[1]), 
            nn.ReLU(), 
            nn.Dropout(encoder_dropout_rates[1]),
            
            nn.Linear(encoder_dims[1], hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(encoder_dropout_rates[2])
        ]
        self.instance_encoder = nn.Sequential(*encoder_layers).to(self.device_model)
        
        if self.attention_type in ['linformer', 'linear']:
            self.num_heads = model_params['num_heads']
            self.head_dim = hidden_dim // self.num_heads
            self.q_proj = nn.Linear(hidden_dim, hidden_dim).to(device_model)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim).to(device_model)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim).to(device_model)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim).to(device_model)
            self.dropout = nn.Dropout(model_params['linformer_dropout'])
            if self.attention_type == 'linformer':
                self.linformer_k = model_params['linformer_k']
                self.E_proj = nn.Linear(10000, self.linformer_k, bias=False).to(device_model)
                self.F_proj = nn.Linear(10000, self.linformer_k, bias=False).to(device_model)
            elif self.attention_type == 'linear':
                self.feature_map = nn.ReLU()
        elif self.attention_type == 'standard': # Standard MHA
            self.instance_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=model_params['num_heads'],
                dropout=model_params['mha_dropout'], batch_first=True
            ).to(device_encoder)

   
        if self.attention_type != 'no':
            self.norm1 = nn.LayerNorm(hidden_dim).to(device_model)
            self.norm2 = nn.LayerNorm(hidden_dim).to(device_model)
        
        self.instance_dropout = InstanceDropout(dropout_rate=model_params['instance_dropout_rate'])
        
        if self.use_moe:
            self.mil_aggregator = MoEMILAggregation(
                input_dim=hidden_dim, hidden_dim=model_params['classifier_dims'][0], 
                num_experts=model_params['moe_num_experts'],
                dropout_rate=model_params['moe_dropout']
            ).to(device_model)
        else:
            self.mil_aggregator = AdaptiveMILAggregation(
                input_dim=hidden_dim, hidden_dim=model_params['classifier_dims'][0]
            ).to(device_model)
        
        classifier_dims = model_params['classifier_dims']
        n_classes = model_params['n_classes']
        classifier_dropout_rates = model_params['classifier_dropout_rates']  

        classifier_layers = [
            nn.Linear(hidden_dim, classifier_dims[0]),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_rates[0]),
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_rates[1]),
            nn.Linear(classifier_dims[1], n_classes)
        ]
        self.classifier = nn.Sequential(*classifier_layers).to(self.device_model)
        
        self.use_domain_adaptation = ablation_params.get('use_domain_adaptation', True)
        if self.use_domain_adaptation:
            self.gradient_reversal = GradientReversalLayer().to(device_model)
            self.domain_classifier = DomainClassifier(
                input_dim=hidden_dim, 
                hidden_dim=model_params['classifier_dims'][0], 
                num_domains=num_domains
            ).to(device_model)

        initialize_weights(self)

    def process_large_bag(self, instances):
        if instances.size(0) <= self.max_instances:
            return self.instance_encoder(instances)
        chunk_features = [self.instance_encoder(chunk) for chunk in torch.split(instances, self.max_instances)]
        return torch.cat(chunk_features, dim=0)

    def linformer_attention_forward(self, x):
        _, seq_len, embed_dim = 1, x.size(0), x.size(1)
        Q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        
        if seq_len > self.linformer_k:
            K_t = K.transpose(1, 2)
            V_t = V.transpose(1, 2)
            if seq_len <= self.E_proj.weight.size(0):
                 E_proj_matrix = self.E_proj.weight[:seq_len, :self.linformer_k]
                 F_proj_matrix = self.F_proj.weight[:seq_len, :self.linformer_k]
            else:
                E_proj_matrix = F.interpolate(self.E_proj.weight.T.unsqueeze(0).unsqueeze(0), size=(self.linformer_k, seq_len), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).T
                F_proj_matrix = F.interpolate(self.F_proj.weight.T.unsqueeze(0).unsqueeze(0), size=(self.linformer_k, seq_len), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).T
            
            K = torch.matmul(K_t, E_proj_matrix)  # [num_heads, head_dim, linformer_k]
            V = torch.matmul(V_t, F_proj_matrix)
            K = K.transpose(1, 2)  # [num_heads, linformer_k, head_dim]
            V = V.transpose(1, 2)
            
        attn_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), V)
        return self.out_proj(attn_output.transpose(0, 1).contiguous().view(seq_len, embed_dim))

    def linear_attention_forward(self, x):
        seq_len, embed_dim = x.size(0), x.size(1)
        Q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        
        Q_mapped = self.feature_map(Q)
        K_mapped = self.feature_map(K)
        
        KV_mult = torch.matmul(K_mapped.transpose(-2, -1), V)
        attn_output = torch.matmul(Q_mapped, KV_mult)
        
        normalizer = torch.matmul(Q_mapped, K_mapped.sum(dim=1, keepdim=True).transpose(-2, -1))
        attn_output = attn_output / (normalizer + 1e-6)
        
        return self.out_proj(self.dropout(attn_output).transpose(0, 1).contiguous().view(seq_len, embed_dim))

    def forward(self, x, alpha=1.0):
        if x.is_sparse: 
            x = x.to_dense()
            
        instance_features = self.process_large_bag(x)
        instance_features = self.instance_dropout(instance_features)
        
        if self.attention_type == 'linformer':
            attended_instances = self.linformer_attention_forward(instance_features)
            attended_instances = self.norm1(attended_instances + instance_features)
            attended_instances = self.norm2(attended_instances)
        elif self.attention_type == 'linear':
            attended_instances = self.linear_attention_forward(instance_features)
            attended_instances = self.norm1(attended_instances + instance_features)
            attended_instances = self.norm2(attended_instances)
        elif self.attention_type == 'standard':
            chunk_size = 1024
            instance_chunks = torch.split(instance_features, chunk_size, dim=0)
            attended_chunks = []

            for chunk in instance_chunks:
                chunk_for_attn = chunk.to(self.device_encoder)
                attended_chunk, _ = self.instance_attention(
                    chunk_for_attn.unsqueeze(0),
                    chunk_for_attn.unsqueeze(0),
                    chunk_for_attn.unsqueeze(0)
                )
                attended_chunks.append(attended_chunk.squeeze(0).to(self.device_model))

            attended_instances = torch.cat(attended_chunks, dim=0)
            attended_instances = self.norm1(attended_instances + instance_features)
            attended_instances = self.norm2(attended_instances)
        else:  # attention_type == 'no'
            attended_instances = instance_features
            
        bag_features, attention_weights = self.mil_aggregator(attended_instances)
        disease_output = self.classifier(bag_features.unsqueeze(0)).squeeze(0)
        
        domain_output = None
        if self.use_domain_adaptation:
            self.gradient_reversal.set_lambda(alpha)
            domain_features = self.gradient_reversal(bag_features)
            domain_output = self.domain_classifier(domain_features.unsqueeze(0)).squeeze(0)
            
        return bag_features, disease_output, domain_output, attention_weights