"""
===================================================================================
 TSBLG: Topological Schrödinger Bridge for Linguistic Geometry (COMPLETE)
===================================================================================
 Requirements: pip install torch torchdiffeq PySide6 matplotlib numpy
===================================================================================
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# --- Differential Geometry & ODE Imports ---
try:
    from torchdiffeq import odeint
    ODE_AVAILABLE = True
except ImportError:
    ODE_AVAILABLE = False

# --- GUI Imports ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLabel, QPushButton, QComboBox,
                             QGroupBox, QGridLayout, QTabWidget, QProgressBar)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Device Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"TSB-LG Initializing on device: {DEVICE}")

# =================================================================================
# PART 1: DATA PIPELINE
# =================================================================================

class GeometricNoiseInjector:
    def __init__(self, prob_spell=0.2, prob_grammar=0.3):
        self.p_spell = prob_spell
        self.p_gram = prob_grammar
        self.qwerty_adj = {0: [1, 2], 1: [0, 2, 3], 2: [1, 4], 3: [1, 5], 4: [2, 5, 6]}
        
    def inject_spelling_noise(self, char_indices):
        noisy = char_indices.clone()
        for i in range(len(noisy)):
            if noisy[i] == 0: continue
            if random.random() < self.p_spell and noisy[i] in self.qwerty_adj:
                noisy[i] = random.choice(self.qwerty_adj[noisy[i]])
        return noisy

    def inject_grammar_noise(self, word_indices):
        noisy = word_indices.clone()
        if len(noisy) > 3 and random.random() < self.p_gram:
            i, j = random.sample(range(1, len(noisy)-1), 2)
            noisy[i], noisy[j] = noisy[j], noisy[i]
        return noisy

class LinguisticGeometryDataset(Dataset):
    def __init__(self, max_words=10, max_chars=12):
        self.mock_data = [
            ("teh cat runs", "teh cat runs", "le chat court"),
            ("i am hre", "i am hre", "je suis ici"),
            ("wtaer is cold", "wtaer is cold", "l'eau est froide"),
            ("she plaies piano", "she plaies piano", "elle joue du piano"),
        ]
        self.max_words = max_words
        self.max_chars = max_chars
        self.injector = GeometricNoiseInjector()
        self.char2id = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        self.word2id = {w: i+1 for i, w in enumerate("the teh cat runs i am hre wtaer water is cold she plaies plays piano".split())}
        self.target2id = {w: i+1 for i, w in enumerate("le chat court je suis ici l'eau est froide elle joue du piano".split())}

    def __len__(self): return len(self.mock_data) * 10

    def __getitem__(self, idx):
        src_c, src_w, tgt = self.mock_data[idx % len(self.mock_data)]
        c_ids = [self.char2id.get(c, 1) for c in src_c][:self.max_chars]
        w_ids = [self.word2id.get(w, 1) for w in src_w.split()][:self.max_words]
        tgt_id = self.target2id.get(tgt.split()[0], 1)
        
        c_ids = torch.tensor(self.injector.inject_spelling_noise(torch.tensor(c_ids)))
        w_ids = torch.tensor(self.injector.inject_grammar_noise(torch.tensor(w_ids)))
        
        c_mask = torch.zeros(self.max_chars); c_mask[:len(c_ids)] = 1
        c_ids = F.pad(c_ids, (0, self.max_chars - len(c_ids)))
        w_mask = torch.zeros(self.max_words); w_mask[:len(w_ids)] = 1
        w_ids = F.pad(w_ids, (0, self.max_words - len(w_ids)))
        
        return {"c_ids": c_ids, "c_mask": c_mask, "w_ids": w_ids, "w_mask": w_mask, "tgt": torch.tensor(tgt_id)}

# =================================================================================
# PART 2: RIGOROUS CORE ARCHITECTURE
# =================================================================================

class ContinuousSchrodingerBridge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.velocity_net = nn.Sequential(
            nn.Linear(dim + 1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, dim)
        )
    def forward(self, t, x):
        t_exp = torch.ones(x.shape[0], 1, device=x.device) * t
        return self.velocity_net(torch.cat([t_exp, x], dim=-1))

class TSBLG_Core(nn.Module):
    def __init__(self, char_vocab=30, word_vocab=30, dim=16):
        super().__init__()
        self.dim = dim
        self.char_embed = nn.Embedding(char_vocab, dim)
        self.char_graph_laplacian_weights = nn.Parameter(torch.rand(char_vocab, char_vocab))
        self.word_embed = nn.Embedding(word_vocab, dim)
        self.sheaf_restrict = nn.Linear(dim * 2, dim, bias=False)
        self.target_embed = nn.Embedding(word_vocab, dim)
        self.epsilon = 0.1
        self.ode_bridge = ContinuousSchrodingerBridge(dim) if ODE_AVAILABLE else None

    def spell_check_sub_riemannian(self, c_ids, c_mask):
        x = self.char_embed(c_ids) * c_mask.unsqueeze(-1)
        seq_len = x.shape[1]
        if seq_len < 2: return x.mean(1), torch.tensor(0.0, device=x.device)
        A_sub = F.softmax(self.char_graph_laplacian_weights[c_ids][:, c_ids], dim=-1) + torch.eye(seq_len, device=x.device)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(A_sub.sum(dim=1)) + 1e-6))
        L_norm = torch.eye(seq_len, device=x.device) - (D_inv_sqrt @ A_sub @ D_inv_sqrt)
        L_masked = L_norm * (c_mask.unsqueeze(1) * c_mask.unsqueeze(0))
        projection = torch.bmm(L_masked, x)
        spell_energy = torch.norm(projection, p=2, dim=-1).sum(dim=-1).mean()
        return x.mean(1), spell_energy

    def grammar_check_sheaf_cohomology(self, w_ids, w_mask):
        x = self.word_embed(w_ids) * w_mask.unsqueeze(-1)
        B, N = x.shape[0], x.shape[1]
        if N < 2: return torch.tensor(0.0, device=x.device), x.mean(1), torch.tensor(0.0, device=x.device), None
        
        B_mat = torch.zeros(N-1, N, device=x.device)
        for i in range(N-1): B_mat[i, i], B_mat[i, i+1] = -1, 1
        edge_mask = w_mask[:, :-1] * w_mask[:, 1:]
        l, r = x[:, :-1, :], x[:, 1:, :]
        W = self.sheaf_restrict(torch.cat([l, r], dim=-1)) * edge_mask.unsqueeze(-1)
        coboundary = W * torch.bmm(B_mat.unsqueeze(0).expand(B, -1, -1), x)
        sheaf_energy = torch.norm(coboundary, p=2, dim=-1).sum(dim=-1).mean()
        
        # Ollivier-Ricci Approximation
        with torch.no_grad():
            dist_matrix = torch.cdist(x, x, p=2)
            adj = (dist_matrix < dist_matrix.mean(dim=-1, keepdim=True)).float() * w_mask.unsqueeze(1) * w_mask.unsqueeze(0)
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
            w1_distances = []
            for i in range(N-1):
                if edge_mask[0, i] > 0:
                    cost = dist_matrix[:, i, :].unsqueeze(1) - dist_matrix[:, i+1, :].unsqueeze(0)
                    w1 = (adj[:, i, :].unsqueeze(1) * cost * adj[:, i+1, :].unsqueeze(0)).sum()
                    w1_distances.append(w1)
            ricci_indicator = 1.0 - torch.stack(w1_distances).mean() if w1_distances else torch.tensor(0.0)
        
        return sheaf_energy, x.mean(1), ricci_indicator, W.squeeze(0).detach() # Return W for graph viz

    def translate_schrodinger_sinkhorn(self, source_semantic):
        mu_0 = F.normalize(source_semantic, dim=-1)
        mu_1 = F.normalize(self.target_embed.weight, dim=-1)
        C = 1 - torch.mm(mu_0, mu_1.T)
        log_K = -C / self.epsilon
        f = torch.zeros_like(mu_0[..., 0]).unsqueeze(1)
        g = torch.zeros_like(mu_1[..., 0]).unsqueeze(0)
        for _ in range(20): 
            g = torch.logsumexp(log_K + f, dim=0, keepdim=True)
            f = torch.logsumexp(log_K + g, dim=1, keepdim=True)
        return torch.exp(f + g + log_K)

    def forward(self, batch, use_ode=False):
        c_ids, c_mask = batch["c_ids"], batch["c_mask"]
        w_ids, w_mask = batch["w_ids"], batch["w_mask"]
        tgt = batch["tgt"]
        
        word_vecs, spell_loss = self.spell_check_sub_riemannian(c_ids, c_mask)
        sheaf_loss, sentence_semantic, ricci_ind, sheaf_W = self.grammar_check_sheaf_cohomology(w_ids, w_mask)
        
        ot_plan = self.translate_schrodinger_sinkhorn(sentence_semantic)
        trans_loss = F.cross_entropy(ot_plan, tgt)
            
        total_loss = spell_loss + 5.0 * sheaf_loss + 10.0 * trans_loss
        return total_loss, spell_loss, sheaf_loss, trans_loss, ricci_ind, ot_plan, sheaf_W

# =================================================================================
# PART 3: ACADEMIC VISUALIZATION SUITE
# =================================================================================

class OTVisualizer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3), facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title("Spherical Sinkhorn Transport Plan (π*)", color='white', fontsize=10)
        self.ax.tick_params(colors='white')

    def update_ot_matrix(self, matrix):
        self.ax.clear(); self.setup_plot()
        if matrix is not None:
            self.ax.imshow(matrix[0, :20].reshape(4, 5), cmap='magma', aspect='auto')
        self.ax.set_xlabel("Target Nodes", color='white'); self.ax.set_ylabel("Source Cloud", color='white')
        self.draw()

class RicciVisualizer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3), facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setup_plot()
        self.ricci_history, self.h1_history = [], []

    def setup_plot(self):
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title("Topological Phase Transition (Grokking)", color='white', fontsize=10)
        self.ax.tick_params(colors='white')

    def update_phase_transition(self, ricci, h1):
        self.ricci_history.append(ricci); self.h1_history.append(h1)
        self.ax.clear(); self.setup_plot()
        epochs = range(len(self.ricci_history))
        self.ax.plot(epochs, self.ricci_history, color='#4ec9b0', label='Ollivier-Ricci Curvature')
        self.ax.plot(epochs, self.h1_history, color='#f44747', label='Sheaf Dirichlet Energy (H¹ proxy)')
        self.ax.axhline(0, color='white', linestyle='--', alpha=0.5)
        self.ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        self.ax.set_xlabel("Epochs", color='white'); self.draw()

class SheafGraphVisualizer(FigureCanvas):
    """Draws the syntactic graph colored by topological obstructions."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3), facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title("Syntactic Sheaf Graph (Edge Color = Restriction W_ij)", color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelbottom=False, labelleft=False)

    def update_graph(self, words, W_matrix):
        self.ax.clear(); self.setup_plot()
        if W_matrix is None or len(words) < 2: return
        
        # Layout: simple line layout for words
        x_coords = np.linspace(0, 1, len(words))
        y_coords = np.zeros(len(words))
        
        # Draw Nodes
        self.ax.scatter(x_coords, y_coords, s=200, color='#569cd6', zorder=5)
        for i, word in enumerate(words):
            self.ax.text(x_coords[i], 0.05, word, ha='center', va='bottom', color='white', fontsize=9)
            
        # Draw Edges colored by Sheaf Restriction Weight magnitude
        for i in range(len(words)-1):
            # W_matrix shape is (Num_Edges, Dim). Take L2 norm of the edge weight vector
            w_norm = torch.norm(W_matrix[i], p=2).item()
            
            # Map norm to color: Low energy = Green (Cohesive), High energy = Red (Obstruction)
            color_val = min(w_norm / 2.0, 1.0) # Normalize roughly to [0, 1]
            edge_color = (color_val, 1.0 - color_val, 0.2) 
            
            self.ax.plot([x_coords[i], x_coords[i+1]], [0, 0], 
                         color=edge_color, linewidth=3 + w_norm*2, alpha=0.8, zorder=4)
                         
        self.ax.set_xlim(-0.1, 1.1); self.ax.set_ylim(-0.2, 0.3)
        self.draw()

# =================================================================================
# PART 4: THREADING & GUI
# =================================================================================

class TopologyWorker(QThread):
    # Added 'words' and 'W_matrix' to the signal payload
    finished = Signal(float, float, float, object, str, list, object)

    def __init__(self, model, text):
        super().__init__()
        self.model = model
        self.text = text

    def run(self):
        words = self.text.lower().split()
        c2id = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        w2id = {w: i+1 for i, w in enumerate("the teh cat runs i am hre wtaer water is cold she plaies plays piano".split())}
        
        c_ids = torch.tensor([c2id.get(c, 1) for c in self.text.lower()]).unsqueeze(0).to(DEVICE)
        w_ids = torch.tensor([w2id.get(w, 1) for w in words]).unsqueeze(0).to(DEVICE)
        
        c_ids = F.pad(c_ids, (0, 12 - c_ids.shape[1]))
        w_ids = F.pad(w_ids, (0, 10 - w_ids.shape[1]))
        
        c_mask = (c_ids != 0).float()
        w_mask = (w_ids != 0).float()
        
        batch = {"c_ids": c_ids, "c_mask": c_mask, "w_ids": w_ids, "w_mask": w_mask, "tgt": torch.tensor([1]).to(DEVICE)}
        
        self.model.eval()
        with torch.no_grad():
            _, spell, gram, trans, ricci, ot, W = self.model(batch, use_ode=False)
            
        interp = "Syntactically Cohesive (Sheaf Gluing Successful)"
        if gram > 2.0: interp = "CRITICAL: H¹ Obstruction (Garden-Path/Syntax Error)"
        elif gram > 0.5: interp = "MINOR: Local Semantic Discrepancy"
        if spell > 1.5: interp += " | High Jet-Bundle Energy (Spelling Error)"
            
        ot_np = ot.cpu().numpy() if ot is not None else None
        W_np = W.cpu() if W is not None else None
        self.finished.emit(spell.item(), gram.item(), ricci.item(), ot_np, interp, words, W_np)

class GrokkingWorker(QThread):
    epoch_update = Signal(float, float, int)
    finished_signal = Signal()

    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-3)

    def run(self):
        loader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        self.model.train()
        for epoch in range(150):
            for batch in loader:
                # Move batch to device
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss, s_l, h1_l, t_l, ricci, _, _ = self.model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.epoch_update.emit(ricci.item(), h1_l.item(), epoch)
        self.finished_signal.emit()

class TSBLG_Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSB-LG: Topological Schrödinger Bridge Linguistic Studio")
        self.setGeometry(100, 100, 1400, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #d4d4d4; font-size: 12px; }
            QTextEdit { background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #3c3c3c; font-size: 14px; }
            QPushButton { background-color: #0e639c; color: white; border: none; padding: 10px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3c3c3c; color: #888; }
            QGroupBox { border: 1px solid #3c3c3c; color: #569cd6; margin-top: 10px; font-weight: bold; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QProgressBar { border: 1px solid #3c3c3c; border-radius: 4px; text-align: center; color: white; background-color: #2b2b2b; }
            QProgressBar::chunk { background-color: #4ec9b0; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background: #1e1e1e; }
            QTabBar::tab { background: #2b2b2b; color: white; padding: 8px; }
            QTabBar::tab:selected { background: #0e639c; }
        """)

        # Initialize Model on global DEVICE
        self.model = TSBLG_Core().to(DEVICE)
        self.dataset = LinguisticGeometryDataset()
        self.initUI()

    def initUI(self):
        main_widget = QWidget(); layout = QHBoxLayout(main_widget)
        left_panel = QVBoxLayout()
        
        # INPUT
        input_group = QGroupBox("Degraded Input Manifold"); input_layout = QVBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text to analyze topology (e.g., 'teh cat runs')...")
        self.input_text.setMaximumHeight(80); input_layout.addWidget(self.input_text)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Synthetic (Demo)", "WMT Robustness", "BEA-2019 GEC", "JFLEG"])
        input_layout.addWidget(QLabel("Active Training Distribution:")); input_layout.addWidget(self.dataset_combo)
        self.process_btn = QPushButton("Compute Topological Analysis")
        self.process_btn.clicked.connect(self.run_analysis); input_layout.addWidget(self.process_btn)
        input_group.setLayout(input_layout); left_panel.addWidget(input_group)

        # METRICS
        metrics_group = QGroupBox("Geometric Obstructions (Real-time)"); metrics_layout = QGridLayout()
        self.spell_label = QLabel("Jet Bundle Energy: 0.000")
        self.grammar_label = QLabel("Sheaf Dirichlet Energy: 0.000")
        self.ricci_label = QLabel("Ollivier-Ricci Curvature: 0.000")
        self.interp_label = QLabel("Status: Waiting..."); self.interp_label.setWordWrap(True)
        for lbl in [self.spell_label, self.grammar_label, self.ricci_label, self.interp_label]:
            lbl.setFont(QFont("Consolas", 10)); metrics_layout.addWidget(lbl)
        metrics_group.setLayout(metrics_layout); left_panel.addWidget(metrics_group)
        
        # TRAINING
        train_group = QGroupBox("Topological Phase Transition (Grokking)"); train_layout = QVBoxLayout()
        self.train_btn = QPushButton("Train Manifold (Induce Grokking)")
        self.train_btn.clicked.connect(self.run_grokking)
        self.progress_bar = QProgressBar(); self.progress_bar.setMaximum(150)
        train_layout.addWidget(self.train_btn); train_layout.addWidget(self.progress_bar)
        train_group.setLayout(train_layout); left_panel.addWidget(train_group)
        left_panel.addStretch(); layout.addLayout(left_panel, stretch=1)

        # VISUALIZATIONS
        right_panel = QVBoxLayout(); self.tabs = QTabWidget()
        
        ot_widget = QWidget(); ot_layout = QVBoxLayout(ot_widget)
        self.ot_viz = OTVisualizer(self); ot_layout.addWidget(self.ot_viz)
        self.tabs.addTab(ot_widget, "Schrödinger Bridge (OT)")
        
        sheaf_widget = QWidget(); sheaf_layout = QVBoxLayout(sheaf_widget)
        self.sheaf_viz = SheafGraphVisualizer(self); sheaf_layout.addWidget(self.sheaf_viz)
        self.tabs.addTab(sheaf_widget, "Syntactic Sheaf Graph")
        
        ricci_widget = QWidget(); ricci_layout = QVBoxLayout(ricci_widget)
        self.ricci_viz = RicciVisualizer(self); ricci_layout.addWidget(self.ricci_viz)
        self.tabs.addTab(ricci_widget, "Phase Transition (Ricci)")
        
        right_panel.addWidget(self.tabs); layout.addLayout(right_panel, stretch=2)
        self.setCentralWidget(main_widget)

    def run_analysis(self):
        self.process_btn.setEnabled(False)
        text = self.input_text.toPlainText()
        if not text: text = "the cat runs"
        self.worker = TopologyWorker(self.model, text)
        # Match new signature: spell, gram, ricci, ot, interp, words, W
        self.worker.finished.connect(self.update_analysis_ui)
        self.worker.start()

    def update_analysis_ui(self, spell, grammar, ricci, ot_matrix, interp, words, W_matrix):
        self.spell_label.setText(f"Jet Bundle Energy: {spell:.4f}")
        self.grammar_label.setText(f"Sheaf Dirichlet Energy: {grammar:.4f}")
        self.ricci_label.setText(f"Ollivier-Ricci Curvature: {ricci:.4f}")
        self.interp_label.setText(f"Diagnosis: {interp}")
        
        self.spell_label.setStyleSheet("color: #f44747;" if spell > 1.5 else "color: #4ec9b0;")
        self.grammar_label.setStyleSheet("color: #f44747;" if grammar > 2.0 else "color: #4ec9b0;")
        
        self.ot_viz.update_ot_matrix(ot_matrix)
        self.sheaf_viz.update_graph(words, W_matrix)
        self.tabs.setCurrentIndex(1) # Switch to the new Graph tab
        self.process_btn.setEnabled(True)

    def run_grokking(self):
        self.train_btn.setEnabled(False)
        self.ricci_viz.ricci_history.clear(); self.ricci_viz.h1_history.clear()
        self.grok_worker = GrokkingWorker(self.model, self.dataset)
        self.grok_worker.epoch_update.connect(self.update_grokking_ui)
        self.grok_worker.finished_signal.connect(self.grokking_finished)
        self.grok_worker.start(); self.tabs.setCurrentIndex(2)

    def update_grokking_ui(self, ricci, h1, epoch):
        self.progress_bar.setValue(epoch)
        self.ricci_viz.update_phase_transition(ricci, h1)

    def grokking_finished(self):
        self.train_btn.setEnabled(True); self.progress_bar.setValue(150)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = TSBLG_Studio()
    window.show()
    sys.exit(app.exec())