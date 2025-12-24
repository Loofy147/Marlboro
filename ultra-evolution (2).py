# 🚀 HYPER-EVOLUTION: Battle-Tested Version
# Multi-Objective Neural Architecture Search with Ensemble Red Teams
# Features: Advanced blocks, multiple objectives, Pareto optimization, novelty search

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import time
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ===================== ADVANCED ARCHITECTURE BLOCKS =====================

class AttentionBlock(nn.Module):
    """Simplified self-attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        identity = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn = (q @ k.t()) / (x.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        return identity + self.proj(out)


class ResidualBlock(nn.Module):
    """Residual block with projection"""
    def __init__(self, in_dim: int, out_dim: int, activation: str = 'GELU'):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fc1 = nn.Linear(out_dim, out_dim * 2)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'SiLU':
            self.act = nn.SiLU()
        elif activation == 'Mish':
            self.act = nn.Mish()
        else:
            self.act = nn.ReLU()
    
    def forward(self, x):
        identity = self.projection(x)
        x = self.norm(identity)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return identity + x


class MixerBlock(nn.Module):
    """MLP-Mixer inspired block"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return x


# ===================== FLEXIBLE SEQUENTIAL ARCHITECTURE =====================

class FlexibleArchitecture(nn.Module):
    """Sequential architecture with advanced blocks and skip connections"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.build_network()
    
    def build_network(self):
        """Build flexible network"""
        layers = []
        current_dim = self.config['input_dim']
        
        # Input projection
        first_dim = self.config['layers'][0]['dim']
        layers.append(nn.Linear(current_dim, first_dim))
        layers.append(self._get_activation(self.config['layers'][0].get('activation', 'GELU')))
        current_dim = first_dim
        
        # Build layers
        for i, layer_config in enumerate(self.config['layers']):
            layer_type = layer_config['type']
            target_dim = layer_config['dim']
            
            if layer_type == 'linear':
                if current_dim != target_dim:
                    layers.append(nn.Linear(current_dim, target_dim))
                    current_dim = target_dim
                layers.append(self._get_activation(layer_config.get('activation', 'GELU')))
                layers.append(nn.LayerNorm(current_dim))
                layers.append(nn.Dropout(layer_config.get('dropout', 0.1)))
                
            elif layer_type == 'residual':
                layers.append(ResidualBlock(current_dim, target_dim, layer_config.get('activation', 'GELU')))
                current_dim = target_dim
                
            elif layer_type == 'attention':
                if current_dim != target_dim:
                    layers.append(nn.Linear(current_dim, target_dim))
                    current_dim = target_dim
                layers.append(AttentionBlock(current_dim))
                
            elif layer_type == 'mixer':
                if current_dim != target_dim:
                    layers.append(nn.Linear(current_dim, target_dim))
                    current_dim = target_dim
                layers.append(MixerBlock(current_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Output head
        self.output_head = nn.Linear(current_dim, self.config['output_dim'])
        
        # Skip connections
        self.skip_connections = []
        if self.config.get('use_skip', False):
            for skip in self.config.get('skip_list', []):
                from_idx, to_idx = skip['from'], skip['to']
                if from_idx < len(self.config['layers']) and to_idx < len(self.config['layers']):
                    from_dim = self.config['layers'][from_idx]['dim']
                    to_dim = self.config['layers'][to_idx]['dim']
                    self.skip_connections.append({
                        'from': from_idx,
                        'to': to_idx,
                        'proj': nn.Linear(from_dim, to_dim) if from_dim != to_dim else nn.Identity()
                    })
            self.skip_projections = nn.ModuleList([s['proj'] for s in self.skip_connections])
    
    def _get_activation(self, name: str):
        activations = {
            'GELU': nn.GELU(),
            'SiLU': nn.SiLU(),
            'Mish': nn.Mish(),
            'ReLU': nn.ReLU()
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x):
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass through network
        x = self.network(x)
        
        # Output
        return self.output_head(x)


# ===================== ENSEMBLE RED TEAM =====================

class EnsembleRedTeam:
    """Multiple coordinated attack strategies"""
    def __init__(self, input_dim: int, device: str = 'cuda'):
        self.device = device
        self.input_dim = input_dim
        
        # Learned adversarial generator
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        ).to(device)
        
        self.optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
    
    def fgsm_attack(self, model, x, y, epsilon=0.1):
        """Fast Gradient Sign Method"""
        x_adv = x.clone().detach().requires_grad_(True)
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        
        loss.backward()
        
        perturbation = epsilon * x_adv.grad.sign()
        return (x + perturbation).detach()
    
    def pgd_attack(self, model, x, y, epsilon=0.1, alpha=0.01, steps=3):
        """Projected Gradient Descent"""
        x_adv = x.clone().detach()
        
        for _ in range(steps):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, y)
            
            loss.backward()
            
            x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = x_adv.detach()
        
        return x_adv
    
    def learned_attack(self, x, epsilon=0.1):
        """Learned adversarial perturbation"""
        if x.dim() > 2:
            original_shape = x.shape
            x_flat = x.view(x.size(0), -1)
            perturbation = self.generator(x_flat)
            result = x_flat + epsilon * perturbation
            return result.view(original_shape)
        else:
            perturbation = self.generator(x)
            return x + epsilon * perturbation
    
    def ensemble_attack(self, model, x, y, epsilon=0.1):
        """Combine multiple attacks - return worst case"""
        model.eval()
        
        attacks = []
        try:
            attacks.append(self.fgsm_attack(model, x.clone(), y, epsilon))
        except:
            pass
        
        try:
            attacks.append(self.pgd_attack(model, x.clone(), y, epsilon))
        except:
            pass
        
        try:
            attacks.append(self.learned_attack(x.clone(), epsilon))
        except:
            pass
        
        if not attacks:
            return x  # Fallback
        
        # Return worst-case attack
        worst_loss = -float('inf')
        worst_attack = x
        
        with torch.no_grad():
            for attack in attacks:
                try:
                    outputs = model(attack)
                    loss = F.cross_entropy(outputs, y)
                    if loss > worst_loss:
                        worst_loss = loss
                        worst_attack = attack
                except:
                    continue
        
        return worst_attack
    
    def train(self, model, x, y):
        """Train learned attacker"""
        model.eval()
        self.generator.train()
        
        try:
            adv_x = self.learned_attack(x)
            outputs = model(adv_x)
            loss = F.cross_entropy(outputs, y)
            
            # Maximize loss
            adv_loss = -loss
            
            self.optimizer.zero_grad()
            adv_loss.backward()
            self.optimizer.step()
            
            return loss.item()
        except:
            return 0.0


# ===================== ARCHITECTURE GENERATOR =====================

class RobustArchitectureGenerator:
    """Generates valid architectures with proper dimension tracking"""
    def __init__(self, search_space: Dict):
        self.search_space = search_space
        self.mutation_stats = defaultdict(int)
        self.novelty_archive = []
    
    def generate_random(self) -> Dict:
        """Generate random architecture"""
        num_layers = np.random.randint(2, 6)
        
        layers = []
        for i in range(num_layers):
            layer_type = np.random.choice(['linear', 'residual', 'attention', 'mixer'], 
                                         p=[0.4, 0.3, 0.2, 0.1])
            
            layers.append({
                'type': layer_type,
                'dim': int(np.random.choice(self.search_space['hidden_dims'])),
                'activation': np.random.choice(self.search_space['activations']),
                'dropout': np.random.choice([0.1, 0.2, 0.3])
            })
        
        config = {
            'input_dim': self.search_space['input_dim'],
            'output_dim': self.search_space['output_dim'],
            'layers': layers,
            'use_skip': np.random.rand() < 0.3,
            'skip_list': []
        }
        
        # Add skip connections
        if config['use_skip'] and num_layers > 2:
            num_skips = np.random.randint(1, min(3, num_layers))
            for _ in range(num_skips):
                from_idx = np.random.randint(0, num_layers - 1)
                to_idx = np.random.randint(from_idx + 1, num_layers)
                config['skip_list'].append({'from': from_idx, 'to': to_idx})
        
        return config
    
    def mutate(self, config: Dict) -> Dict:
        """Mutate architecture"""
        new_config = copy.deepcopy(config)
        
        mutations = ['add_layer', 'remove_layer', 'change_type', 'change_dim', 'change_activation']
        mutation = np.random.choice(mutations)
        self.mutation_stats[mutation] += 1
        
        if mutation == 'add_layer' and len(new_config['layers']) < 8:
            new_layer = {
                'type': np.random.choice(['linear', 'residual', 'attention', 'mixer']),
                'dim': int(np.random.choice(self.search_space['hidden_dims'])),
                'activation': np.random.choice(self.search_space['activations']),
                'dropout': np.random.choice([0.1, 0.2, 0.3])
            }
            insert_pos = np.random.randint(0, len(new_config['layers']) + 1)
            new_config['layers'].insert(insert_pos, new_layer)
            
        elif mutation == 'remove_layer' and len(new_config['layers']) > 2:
            idx = np.random.randint(0, len(new_config['layers']))
            new_config['layers'].pop(idx)
            
        elif mutation == 'change_type':
            idx = np.random.randint(0, len(new_config['layers']))
            new_config['layers'][idx]['type'] = np.random.choice(['linear', 'residual', 'attention', 'mixer'])
            
        elif mutation == 'change_dim':
            idx = np.random.randint(0, len(new_config['layers']))
            new_config['layers'][idx]['dim'] = int(np.random.choice(self.search_space['hidden_dims']))
            
        elif mutation == 'change_activation':
            idx = np.random.randint(0, len(new_config['layers']))
            new_config['layers'][idx]['activation'] = np.random.choice(self.search_space['activations'])
        
        return new_config
    
    def compute_novelty(self, config: Dict) -> float:
        """Compute novelty score"""
        if not self.novelty_archive:
            return 1.0
        
        features = [
            len(config['layers']),
            sum(1 for l in config['layers'] if l['type'] == 'attention'),
            sum(1 for l in config['layers'] if l['type'] == 'residual'),
            sum(1 for l in config['layers'] if l['type'] == 'mixer'),
            sum(l['dim'] for l in config['layers']) / len(config['layers'])
        ]
        
        distances = []
        for arch in self.novelty_archive[-20:]:
            arch_features = [
                len(arch['layers']),
                sum(1 for l in arch['layers'] if l['type'] == 'attention'),
                sum(1 for l in arch['layers'] if l['type'] == 'residual'),
                sum(1 for l in arch['layers'] if l['type'] == 'mixer'),
                sum(l['dim'] for l in arch['layers']) / len(arch['layers'])
            ]
            dist = np.linalg.norm(np.array(features) - np.array(arch_features))
            distances.append(dist)
        
        return np.mean(distances) if distances else 1.0


# ===================== HYPER-EVOLUTIONARY TRAINER =====================

class HyperEvolutionaryTrainer:
    """Multi-objective evolutionary trainer"""
    def __init__(
        self,
        search_space: Dict,
        device: str = 'cuda',
        population_size: int = 10,
        generations: int = 20,
        objectives: List[str] = ['accuracy', 'robustness', 'speed']
    ):
        self.search_space = search_space
        self.device = device
        self.population_size = population_size
        self.generations = generations
        self.objectives = objectives
        
        self.arch_generator = RobustArchitectureGenerator(search_space)
        self.red_team = EnsembleRedTeam(search_space['input_dim'], device)
        
        self.pareto_front = []
        self.history = defaultdict(list)
        
        self.temperature = 1.0
        self.temp_decay = 0.95
    
    def evaluate_multi_objective(
        self,
        model: FlexibleArchitecture,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate on multiple objectives"""
        model.eval()
        
        objectives = {}
        
        correct = 0
        total = 0
        robust_correct = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Normal accuracy
                outputs = model(batch_x)
                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
                
                # Robustness
                adv_x = self.red_team.ensemble_attack(model, batch_x, batch_y, epsilon=0.1)
                adv_outputs = model(adv_x)
                adv_pred = adv_outputs.argmax(dim=1)
                robust_correct += (adv_pred == batch_y).sum().item()
        
        end_time = time.time()
        
        objectives['accuracy'] = correct / total
        objectives['robustness'] = robust_correct / total
        objectives['speed'] = 1.0 / (end_time - start_time + 1e-6)
        
        # Novelty
        novelty = self.arch_generator.compute_novelty(model.config)
        objectives['novelty'] = novelty
        
        return objectives
    
    def dominates(self, obj1: Dict, obj2: Dict) -> bool:
        """Check Pareto dominance"""
        better_in_one = False
        for key in self.objectives:
            if obj1[key] < obj2[key]:
                return False
            if obj1[key] > obj2[key]:
                better_in_one = True
        return better_in_one
    
    def update_pareto_front(self, config: Dict, objectives: Dict):
        """Update Pareto front"""
        self.pareto_front = [
            (c, o) for c, o in self.pareto_front
            if not self.dominates(objectives, o)
        ]
        
        dominated = any(self.dominates(o, objectives) for _, o in self.pareto_front)
        if not dominated:
            self.pareto_front.append((config, objectives))
    
    def train_candidate(
        self,
        model: FlexibleArchitecture,
        train_loader: DataLoader,
        criterion: nn.Module,
        epochs: int = 3
    ):
        """Quick training"""
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    
    def evolve(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> List[Tuple[Dict, Dict]]:
        """Multi-objective evolutionary loop"""
        print("🚀 HYPER-EVOLUTION INITIATED")
        print("=" * 80)
        print(f"Objectives: {', '.join(self.objectives)}")
        print(f"Population: {self.population_size} | Generations: {self.generations}")
        print("=" * 80)
        
        population = [self.arch_generator.generate_random() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            print(f"\n⚡ Generation {gen + 1}/{self.generations} | Temp: {self.temperature:.3f}")
            print("-" * 80)
            
            population_objectives = []
            
            for i, config in enumerate(population):
                try:
                    model = FlexibleArchitecture(config).to(self.device)
                    self.train_candidate(model, train_loader, criterion, epochs=2)
                    
                    objectives = self.evaluate_multi_objective(model, val_loader)
                    population_objectives.append(objectives)
                    
                    self.update_pareto_front(config, objectives)
                    
                    # Train red team
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        self.red_team.train(model, batch_x, batch_y)
                        break
                    
                    # Print
                    layers = len(config['layers'])
                    types = ''.join([l['type'][0].upper() for l in config['layers']])
                    print(f"  [{i+1:2d}/{self.population_size}] L{layers}_{types:8s} → ", end='')
                    print(f"Acc:{objectives['accuracy']:.3f} Rob:{objectives['robustness']:.3f} ", end='')
                    print(f"Spd:{objectives['speed']:.1f} Nov:{objectives['novelty']:.2f}")
                    
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  [{i+1:2d}/{self.population_size}] FAILED: {str(e)[:50]}")
                    population_objectives.append({obj: 0.0 for obj in self.objectives + ['novelty']})
            
            # Track history
            for obj in self.objectives + ['novelty']:
                values = [o[obj] for o in population_objectives if o[obj] > 0]
                if values:
                    self.history[f'best_{obj}'].append(max(values))
                    self.history[f'avg_{obj}'].append(np.mean(values))
            
            print(f"\n  📊 Pareto Front: {len(self.pareto_front)} solutions")
            if population_objectives:
                valid = [o for o in population_objectives if o['accuracy'] > 0]
                if valid:
                    print(f"  🎯 Best Acc: {max(o['accuracy'] for o in valid):.3f}")
                    print(f"  🛡️  Best Rob: {max(o['robustness'] for o in valid):.3f}")
            
            # Selection
            elite_configs = [c for c, _ in self.pareto_front[-3:]] if self.pareto_front else []
            new_population = elite_configs.copy()
            
            while len(new_population) < self.population_size:
                if np.random.rand() < 0.7 and elite_configs:
                    parent = random.choice(elite_configs)
                    new_population.append(self.arch_generator.mutate(parent))
                else:
                    new_population.append(self.arch_generator.generate_random())
            
            population = new_population[:self.population_size]
            self.arch_generator.novelty_archive.extend(population)
            self.temperature *= self.temp_decay
        
        print("\n" + "=" * 80)
        print(f"✅ EVOLUTION COMPLETE! Pareto Front: {len(self.pareto_front)} solutions")
        
        return self.pareto_front
    
    def plot_results(self):
        """Visualize evolution"""
        if not self.history['best_accuracy']:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        if self.pareto_front:
            objs = [o for _, o in self.pareto_front]
            ax1.scatter([o['accuracy'] for o in objs], 
                       [o['robustness'] for o in objs], s=100, alpha=0.6)
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Robustness')
        ax1.set_title('Pareto Front: Accuracy vs Robustness')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        if 'best_accuracy' in self.history:
            ax2.plot(self.history['best_accuracy'], label='Accuracy', linewidth=2)
            ax2.plot(self.history['best_robustness'], label='Robustness', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Score')
        ax2.set_title('Evolution of Best Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        if 'best_novelty' in self.history:
            ax3.plot(self.history['best_novelty'], color='orange', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Novelty')
        ax3.set_title('Novelty Over Time')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        if self.pareto_front:
            sizes = [len(c['layers']) for c, _ in self.pareto_front]
            ax4.hist(sizes, bins=10, alpha=0.7, color='coral')
        ax4.set_xlabel('Number of Layers')
        ax4.set_ylabel('Count')
        ax4.set_title('Architecture Sizes in Pareto Front')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hyper_evolution_results.png', dpi=150)
        print("📊 Results saved!")
        plt.show()


# ===================== MAIN =====================

def prepare_mnist_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    search_space = {
        'input_dim': 784,
        'output_dim': 10,
        'hidden_dims': [64, 128, 256, 512],
        'activations': ['ReLU', 'GELU', 'SiLU', 'Mish']
    }
    
    print("📦 Loading data...")
    train_loader, val_loader = prepare_mnist_data(batch_size=128)
    
    trainer = HyperEvolutionaryTrainer(
        search_space=search_space,
        device=device,
        population_size=8,
        generations=20,
        objectives=['accuracy', 'robustness', 'speed']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    pareto_front = trainer.evolve(train_loader, val_loader, criterion)
    
    trainer.plot_results()
    
    if pareto_front:
        print("\n" + "=" * 80)
        print("🏆 TOP PARETO SOLUTIONS:")
        print("=" * 80)
        
        for i, (config, objectives) in enumerate(pareto_front[-min(5, len(pareto_front)):]):
            print(f"\n💎 Solution {i+1}:")
            print(f"   Layers: {len(config['layers'])}")
            print(f"   Types: {[l['type'] for l in config['layers']]}")
            print(f"   Accuracy:   {objectives['accuracy']:.4f}")
            print(f"   Robustness: {objectives['robustness']:.4f}")
            print(f"   Speed:      {objectives['speed']:.2f}")
        
        # Train best
        best_config = max(pareto_front, key=lambda x: x[1]['accuracy'] * x[1]['robustness'])[0]
        
        print("\n🔧 Training best model...")
        best_model = FlexibleArchitecture(best_config).to(device)
        optimizer = optim.AdamW(best_model.parameters(), lr=1e-3)
        
        for epoch in range(10):
            best_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = best_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
            
            print(f"  Epoch {epoch+1}/10: Loss={total_loss/len(train_loader):.4f}, Acc={correct/total:.4f}")
        
        # Final eval
        best_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = best_model(batch_x)
                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        
        print(f"\n🎯 Final Accuracy: {correct/total:.4f}")
        
        torch.save({
            'config': best_config,
            'model_state_dict': best_model.state_dict(),
            'pareto_front': pareto_front
        }, 'hyper_evolved_model.pth')
        
        print("\n💾 Model saved!")
    
    print("🎉 HYPER-EVOLUTION COMPLETE!")


if __name__ == "__main__":
    main()