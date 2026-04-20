



# ─────────────────────────────────────────────
# PART 1 – PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight + bias (same initialisation as nn.Linear)
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight
        # Initialised near 0.5 after sigmoid (small positive noise)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform for the weight
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        # Small random perturbation for gate_scores so they break symmetry
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – map gate_scores → gates ∈ (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 – mask the weights (gradients flow through both weight and gates)
        pruned_weights = self.weight * gates

        # Step 3 – standard linear operation
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
      
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu()

    def sparsity_loss(self) -> torch.Tensor:
        
        gates = torch.sigmoid(self.gate_scores)
        return gates.abs().sum()   # gates ≥ 0, so |g| = g

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
   

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten
        return self.layers(x)

    def prunable_layers(self):
        
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def total_sparsity_loss(self) -> torch.Tensor:
        
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.prunable_layers():
            loss = loss + m.sparsity_loss()
        return loss

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        
        all_gates = torch.cat([m.get_gates().flatten() for m in self.prunable_layers()])
        pruned    = (all_gates < threshold).float().sum()
        return (pruned / all_gates.numel()).item()


# ─────────────────────────────────────────────
# PART 3 – Training & Evaluation
# ─────────────────────────────────────────────

def get_data_loaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        # Total Loss = CrossEntropy + λ * SparsityLoss
        cls_loss     = F.cross_entropy(logits, labels)
        sparse_loss  = model.total_sparsity_loss()
        loss         = cls_loss + lam * sparse_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def train_and_evaluate(lam: float, epochs: int = 20, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"  λ = {lam}   |   device = {device}")
    print(f"{'='*55}")

    train_loader, test_loader = get_data_loaders()
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sp  = model.sparsity_level()
            print(f"  Epoch {epoch:3d}  |  loss {avg_loss:.4f}  |"
                  f"  acc {acc*100:.2f}%  |  sparsity {sp*100:.1f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sp  = model.sparsity_level()
    print(f"\n  Final → Accuracy: {final_acc*100:.2f}%  |"
          f"  Sparsity: {final_sp*100:.1f}%")
    return model, final_acc, final_sp


def plot_gate_distribution(model, lam: float, save_path: str = "gate_dist.png"):
    
    all_gates = torch.cat([m.get_gates().flatten() for m in model.prunable_layers()])
    all_gates = all_gates.numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_gates, bins=100, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Gate value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})", fontsize=13)
    ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1.5,
               label='Prune threshold (0.01)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Gate distribution plot saved → {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    LAMBDAS = [1e-5, 1e-4, 5e-4]   # low / medium / high
    EPOCHS  = 20                    # increase for better accuracy

    results = {}
    best_model, best_lam = None, None

    for lam in LAMBDAS:
        model, acc, sp = train_and_evaluate(lam, epochs=EPOCHS)
        results[lam] = {"accuracy": acc, "sparsity": sp, "model": model}
        if best_model is None or sp > results.get(best_lam, {}).get("sparsity", 0):
            best_model, best_lam = model, lam

    # ── Summary table ──────────────────────────────────
    print("\n" + "─"*48)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("─"*48)
    for lam, r in results.items():
        print(f"  {lam:<12} {r['accuracy']*100:>14.2f} {r['sparsity']*100:>14.1f}")
    print("─"*48)

    # ── Gate distribution for best model ───────────────
    plot_gate_distribution(best_model, best_lam, save_path="gate_dist.png")
    print("\nDone.")
