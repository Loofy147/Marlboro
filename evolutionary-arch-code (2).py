# Self-Evolving Architecture Search with Adversarial Fitness
# Ready for Kaggle H100 🚀
# Just run this and watch the magic happen

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import copy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class DynamicArchitecture(nn.Module):
    """Architecture that can be dynamically modified"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.build_network()
    
    def build_network(self):
        """Build network from config"""
        layers = []
        in_dim = self.config['input_dim']
        
        for i in range(self.config['num_layers']):
            out_dim = self.config['hidden_dim']
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Activation
            if self.config['activation'] == 'GELU':
                layers.append(nn.GELU())
            elif self.config['activation'] == 'SiLU':
                layers.append(nn.SiLU())
            elif self.config['activation'] == 'Mish':
                layers.append(nn.Mish())
            else:
                layers.append(nn.ReLU())
            
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, self.config['output_dim']))
        self.network = nn.Sequential(*layers)
        
        # Optional skip connections
        self.use_skip = self.config.get('skip_connections', False)
        if self.use_skip:
            self.skip_projection = nn.Linear(self.config['input_dim'], self.config['hidden_dim'])
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        if self.use_skip and self.config['num_layers'] > 0:
            skip = self.skip_projection(x)
            # Process through network
            out = x
            for i, layer in enumerate(self.network):
                out = layer(out)
                # Add skip connection before final layer
                if i == len(self.network) - 2:
                    out = out + skip
            return out
        return self.network(x)


class RedTeamAdversary(nn.Module):
    """Network that generates adversarial examples to break the main model"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()  # Bounded perturbations
        )
    
    def forward(self, x, epsilon=0.1):
        """Generate adversarial perturbation"""
        # Flatten if needed
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        perturbation = self.generator(x)
        result = x + epsilon * perturbation
        
        # Restore original shape if needed
        if len(original_shape) > 2:
            result = result.view(original_shape)
        
        return result


class ArchitectureGenerator:
    """Generates and mutates architecture configurations"""
    def __init__(self, search_space: Dict):
        self.search_space = search_space
    
    def generate_random_config(self) -> Dict:
        """Generate random architecture configuration"""
        return {
            'input_dim': self.search_space['input_dim'],
            'output_dim': self.search_space['output_dim'],
            'num_layers': np.random.randint(
                self.search_space['num_layers'][0],
                self.search_space['num_layers'][1] + 1
            ),
            'hidden_dim': int(np.random.choice(self.search_space['hidden_dims'])),
            'activation': np.random.choice(self.search_space['activations']),
            'skip_connections': np.random.choice([True, False])
        }
    
    def mutate_config(self, config: Dict) -> Dict:
        """Mutate existing configuration"""
        new_config = copy.deepcopy(config)
        mutation_type = np.random.choice(['layers', 'hidden', 'activation', 'skip'])
        
        if mutation_type == 'layers':
            new_config['num_layers'] = int(np.clip(
                new_config['num_layers'] + np.random.randint(-1, 2),
                self.search_space['num_layers'][0],
                self.search_space['num_layers'][1]
            ))
        elif mutation_type == 'hidden':
            new_config['hidden_dim'] = int(np.random.choice(self.search_space['hidden_dims']))
        elif mutation_type == 'activation':
            new_config['activation'] = np.random.choice(self.search_space['activations'])
        else:
            new_config['skip_connections'] = not new_config['skip_connections']
        
        return new_config


class SelfEvolvingTrainer:
    """Main trainer that coordinates evolution and red teaming"""
    def __init__(
        self,
        search_space: Dict,
        device: str = 'cuda',
        population_size: int = 8,
        generations: int = 20
    ):
        self.search_space = search_space
        self.device = device
        self.population_size = population_size
        self.generations = generations
        
        self.arch_generator = ArchitectureGenerator(search_space)
        self.red_team = RedTeamAdversary(search_space['input_dim']).to(device)
        self.red_team_optimizer = optim.Adam(self.red_team.parameters(), lr=1e-3)
        
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_accuracy': [],
            'best_robustness': []
        }
        
    def evaluate_architecture(
        self,
        model: DynamicArchitecture,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate both accuracy and robustness"""
        model.eval()
        correct = 0
        total = 0
        robust_correct = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Normal accuracy
                outputs = model(batch_x)
                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
                
                # Adversarial robustness
                adv_x = self.red_team(batch_x, epsilon=0.1)
                adv_outputs = model(adv_x)
                adv_pred = adv_outputs.argmax(dim=1)
                robust_correct += (adv_pred == batch_y).sum().item()
        
        accuracy = correct / total
        robustness = robust_correct / total
        
        return accuracy, robustness
    
    def train_red_team(
        self,
        model: DynamicArchitecture,
        dataloader: DataLoader,
        steps: int = 5
    ):
        """Train red team to find weaknesses"""
        model.eval()
        self.red_team.train()
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(steps):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Generate adversarial examples
                adv_x = self.red_team(batch_x, epsilon=0.1)
                
                # Red team wants to MAXIMIZE loss (make the model fail)
                outputs = model(adv_x)
                loss = criterion(outputs, batch_y)
                
                # Red team maximizes loss by minimizing negative loss
                adv_loss = -loss
                
                self.red_team_optimizer.zero_grad()
                adv_loss.backward()
                self.red_team_optimizer.step()
                
                break  # One batch per step
    
    def train_candidate(
        self,
        model: DynamicArchitecture,
        train_loader: DataLoader,
        criterion: nn.Module,
        epochs: int = 3
    ):
        """Quick training for each candidate"""
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def evolve(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict:
        """Main evolution loop"""
        print("🧬 Starting Self-Evolving Architecture Search...")
        print("=" * 70)
        
        # Initialize population
        population = [
            self.arch_generator.generate_random_config()
            for _ in range(self.population_size)
        ]
        
        best_overall = None
        best_fitness = -float('inf')
        
        for gen in range(self.generations):
            print(f"\n🔬 Generation {gen + 1}/{self.generations}")
            print("-" * 70)
            
            fitness_scores = []
            accuracies = []
            robustness_scores = []
            
            for i, config in enumerate(population):
                # Build and train model
                model = DynamicArchitecture(config).to(self.device)
                
                # Train candidate
                self.train_candidate(model, train_loader, criterion, epochs=3)
                
                # Evaluate
                accuracy, robustness = self.evaluate_architecture(model, val_loader, criterion)
                
                # Fitness = weighted combination of accuracy and robustness
                fitness = 0.5 * accuracy + 0.5 * robustness
                fitness_scores.append(fitness)
                accuracies.append(accuracy)
                robustness_scores.append(robustness)
                
                config_str = f"L{config['num_layers']}_H{config['hidden_dim']}_{config['activation'][:4]}"
                print(f"  [{i+1}/{self.population_size}] {config_str:20s} → Acc: {accuracy:.3f} | Rob: {robustness:.3f} | Fit: {fitness:.3f}")
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_overall = config.copy()
                    print(f"      🌟 NEW BEST! (Fitness: {fitness:.3f})")
                
                # Train red team on this model
                self.train_red_team(model, train_loader, steps=3)
                
                # Cleanup
                del model
                torch.cuda.empty_cache()
            
            # Track history
            self.history['best_fitness'].append(max(fitness_scores))
            self.history['avg_fitness'].append(np.mean(fitness_scores))
            self.history['best_accuracy'].append(max(accuracies))
            self.history['best_robustness'].append(max(robustness_scores))
            
            print(f"\n  📊 Gen Stats: Best Fit={max(fitness_scores):.3f} | Avg Fit={np.mean(fitness_scores):.3f}")
            
            # Selection and mutation
            # Select top performers
            top_k = max(2, self.population_size // 3)
            elite_indices = np.argsort(fitness_scores)[-top_k:]
            elite = [population[i] for i in elite_indices]
            
            # Create new population
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                if np.random.rand() < 0.7:  # 70% mutation, 30% random
                    parent = elite[np.random.randint(len(elite))]
                    new_population.append(self.arch_generator.mutate_config(parent))
                else:
                    new_population.append(self.arch_generator.generate_random_config())
            
            population = new_population
        
        print("\n" + "=" * 70)
        print("✅ Evolution Complete!")
        print("\n🏆 Best Architecture Found:")
        print("-" * 70)
        for key, value in best_overall.items():
            print(f"  {key:20s}: {value}")
        print(f"\n  Final Fitness: {best_fitness:.4f}")
        
        return best_overall
    
    def plot_history(self):
        """Plot evolution history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Evolution History', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(self.history['best_fitness'], 'b-', linewidth=2, label='Best')
        axes[0, 0].plot(self.history['avg_fitness'], 'r--', linewidth=2, label='Average')
        axes[0, 0].set_title('Fitness Over Generations')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['best_accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_title('Best Accuracy Over Generations')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['best_robustness'], 'purple', linewidth=2)
        axes[1, 0].set_title('Best Robustness Over Generations')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Robustness')
        axes[1, 0].grid(True, alpha=0.3)
        
        improvement = np.array(self.history['best_fitness'])
        axes[1, 1].plot(improvement, 'orange', linewidth=2)
        axes[1, 1].fill_between(range(len(improvement)), improvement, alpha=0.3)
        axes[1, 1].set_title('Cumulative Best Fitness')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Fitness')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evolution_history.png', dpi=150, bbox_inches='tight')
        print("\n📈 Evolution history saved to 'evolution_history.png'")
        plt.show()


def prepare_mnist_data(batch_size=64):
    """Prepare MNIST dataset"""
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
    """Main execution"""
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Define search space
    search_space = {
        'input_dim': 784,  # MNIST: 28x28 flattened
        'output_dim': 10,  # 10 digit classes
        'num_layers': (2, 6),
        'hidden_dims': [64, 128, 256, 512],
        'activations': ['ReLU', 'GELU', 'SiLU', 'Mish']
    }
    
    # Prepare data
    print("\n📦 Loading MNIST dataset...")
    train_loader, val_loader = prepare_mnist_data(batch_size=128)
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = SelfEvolvingTrainer(
        search_space=search_space,
        device=device,
        population_size=8,  # Increase for better search
        generations=20      # Increase for more evolution
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Run evolution!
    best_config = trainer.evolve(train_loader, val_loader, criterion)
    
    # Plot results
    trainer.plot_history()
    
    # Train and save best model
    print("\n🏗️  Training final best model...")
    final_model = DynamicArchitecture(best_config).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=1e-3)
    
    # Train for more epochs
    final_model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        accuracy = correct / total
        print(f"  Epoch {epoch+1}/10: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")
    
    # Final evaluation
    final_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = final_model(batch_x)
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    
    final_accuracy = correct / total
    print(f"\n🎯 Final Test Accuracy: {final_accuracy:.4f}")
    
    # Save model
    torch.save({
        'config': best_config,
        'model_state_dict': final_model.state_dict(),
        'accuracy': final_accuracy
    }, 'best_evolved_model.pth')
    
    print("\n💾 Best model saved to 'best_evolved_model.pth'")
    print("\n🎉 All done! Your architecture has evolved!")


if __name__ == "__main__":
    main()