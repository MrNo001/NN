"""
Generates the figures for batchnorm_report.md.

Trains small character-level MLPs on names.txt (same setup as
MLP_normaliezed_layers.ipynb) and measures what happens inside the tanh layers.
Run from the repo root:  python batchnorm_figures.py
"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import split_dataset, stoi_and_itos

OUT = 'artifacts'
torch.set_num_threads(1) # tiny matrices: threading overhead costs more than it saves

words = open('names.txt', 'r').read().splitlines()
stoi, itos = stoi_and_itos(words)
block_size = 3
(Xtr, Ytr), (Xdev, Ydev), _ = split_dataset(words, block_size, [0.8, 0.1, 0.1], stoi)

emb_dim = 10
vocab_size = len(stoi)
input_size = block_size * emb_dim
SAT = 0.99 # |tanh| above this -> local gradient (1 - t^2) below 0.02


# ---------------------------------------------------------------- model

def init_mlp(n_layers, n_hidden, init, bn, seed=2147483647):
  """init: 'naive' (W ~ randn), 'kaiming' (gain 5/3) or 'small' (gain 0.5)."""
  g = torch.Generator().manual_seed(seed)
  P = {'C': torch.randn((vocab_size, emb_dim), generator=g), 'layers': []}
  fan_in = input_size
  for _ in range(n_layers):
    W = torch.randn((fan_in, n_hidden), generator=g)
    if init == 'kaiming': W *= (5/3) / fan_in**0.5
    if init == 'small':   W *= 0.5 / fan_in**0.5
    if bn: # bias is redundant before batch norm, bnbias takes its place
      layer = {'W': W, 'gain': torch.ones((1, n_hidden)), 'bias': torch.zeros((1, n_hidden))}
    else:
      layer = {'W': W, 'b': torch.randn(n_hidden, generator=g) * (1 if init == 'naive' else 0.1)}
    P['layers'].append(layer)
    fan_in = n_hidden
  P['Wout'] = torch.randn((fan_in, vocab_size), generator=g) * 0.1 # fix the initial loss
  P['bout'] = torch.zeros(vocab_size)
  P['params'] = [P['C'], P['Wout'], P['bout']] + [t for L in P['layers'] for t in L.values()]
  return P


def forward(P, X, bn_stats=None, capture=None):
  """Returns logits, the input of every tanh and its output.
  bn_stats: fixed (mean, std) per layer for evaluation; capture: list to collect batch stats."""
  x = P['C'][X].view(-1, input_size)
  preacts, acts = [], []
  for i, L in enumerate(P['layers']):
    h = x @ L['W']
    if 'gain' in L:
      if bn_stats is None:
        mean, std = h.mean(0, keepdim=True), h.std(0, keepdim=True)
        if capture is not None: capture.append((mean, std))
      else:
        mean, std = bn_stats[i]
      h = L['gain'] * (h - mean) / std + L['bias']
    else:
      h = h + L['b']
    x = torch.tanh(h)
    preacts.append(h)
    acts.append(x)
  return x @ P['Wout'] + P['bout'], preacts, acts


def saturation(acts):
  return [(a.abs() > SAT).float().mean().item() for a in acts]


@torch.no_grad()
def evaluate(P):
  stats = [] # batch norm: calibrate mean/std on the whole training set
  tr = F.cross_entropy(forward(P, Xtr, capture=stats)[0], Ytr).item()
  dev = F.cross_entropy(forward(P, Xdev, bn_stats=stats or None)[0], Ydev).item()
  return tr, dev


def train(P, steps=30000, batch_size=32, probe_every=500):
  for p in P['params']:
    p.requires_grad = True
  g = torch.Generator().manual_seed(42) # same minibatches for every model
  probe = Xdev[:1000]
  losses, probes = [], []
  for i in range(steps):
    if i % probe_every == 0:
      with torch.no_grad():
        probes.append((i, saturation(forward(P, probe)[2])))

    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    loss = F.cross_entropy(forward(P, Xtr[ix])[0], Ytr[ix])
    for p in P['params']:
      p.grad = None
    loss.backward()
    lr = 0.1 if i < steps // 2 else 0.01
    for p in P['params']:
      p.data += -lr * p.grad
    losses.append(loss.item())
  return {'losses': losses, 'probes': probes, 'eval': evaluate(P)}


# ---------------------------------------------------------------- experiments

def compute():
  torch.manual_seed(0)
  R = {}
  batch = torch.randint(0, Xtr.shape[0], (1000,), generator=torch.Generator().manual_seed(1))

  # 1 hidden layer (the notebook's network) at initialization
  R['shallow_init'] = {}
  for name, init, bn in [('naive', 'naive', False), ('kaiming', 'kaiming', False), ('bn', 'naive', True)]:
    P = init_mlp(1, 200, init, bn)
    with torch.no_grad():
      _, pre, act = forward(P, Xtr[batch])
    R['shallow_init'][name] = {'preact': pre[0], 'act': act[0]}

  # 5 hidden layers at initialization: activations and their gradients
  R['deep_init'] = {}
  for init in ['small', 'kaiming', 'naive']:
    for bn in [False, True]:
      P = init_mlp(5, 100, init, bn)
      for p in P['params']:
        p.requires_grad = True
      logits, pre, act = forward(P, Xtr[batch])
      for a in act:
        a.retain_grad()
      F.cross_entropy(logits, Ytr[batch]).backward()
      R['deep_init'][(init, bn)] = {'act': [a.detach() for a in act], 'grad': [a.grad for a in act]}

  # training runs
  R['runs'] = {}
  for depth, width in [(1, 200), (5, 100)]:
    for name, init, bn in [('naive', 'naive', False), ('kaiming', 'kaiming', False), ('bn', 'naive', True), ('kaiming+bn', 'kaiming', True)]:
      P = init_mlp(depth, width, init, bn)
      R['runs'][(depth, name)] = train(P)
      if depth == 1 and name == 'bn':
        R['bn_model'] = P
      print(depth, name, R['runs'][(depth, name)]['eval'])
  return R


# ---------------------------------------------------------------- figures

INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
COLOR = {'naive': '#eb6834', 'kaiming': '#2a78d6', 'bn': '#1baf7a'}
LABEL = {'naive': 'naive init (W ~ randn)', 'kaiming': 'Kaiming init', 'bn': 'naive init + batch norm'}
LAYER_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b'] # layer 1 (light) -> layer 5 (dark)
FLAT = torch.atanh(torch.tensor(SAT)).item() # |x| > 2.65 is the flat part of tanh

plt.rcParams.update({
  'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
  'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9.5,
  'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
  'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
  'axes.spines.top': False, 'axes.spines.right': False,
  'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
  'axes.titlesize': 10.5, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
  'legend.frameon': False, 'lines.linewidth': 1.8, 'lines.solid_capstyle': 'round',
  'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


def shade_flat(ax, lim):
  for s in (-1, 1):
    ax.axvspan(s * FLAT, s * lim, color=GRID, alpha=0.6, lw=0, zorder=0)


def save(fig, name):
  fig.savefig(f'{OUT}/{name}.png')
  plt.close(fig)


def fig_flat_zone(R):
  S = R['shallow_init']
  lim = 15
  fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={'height_ratios': [1, 1.25]})
  x = torch.linspace(-lim, lim, 600)
  t = torch.tanh(x)
  shade_flat(top, lim)
  shade_flat(bot, lim)
  top.plot(x, t, color=INK)
  top.plot(x, 1 - t**2, color=MUTED)
  top.text(1.5, 0.45, 'tanh(x)', color=INK)
  top.text(-0.8, 0.88, 'local gradient  1 - tanh²(x)', ha='right', color=INK2)
  for s in (-1, 1):
    top.text(s * (FLAT + lim) / 2, -0.55, 'flat zone\ngradient < 0.02', ha='center', color=INK2, fontsize=8.5)
  top.set_ylim(-1.15, 1.15)
  top.set_title('tanh is only useful near zero: outside |x| > 2.65 it passes (almost) no gradient back')

  bins = torch.linspace(-lim, lim, 121)
  for name in ['naive', 'kaiming', 'bn']:
    v = S[name]['preact'].flatten()
    sat = (S[name]['act'].abs() > SAT).float().mean().item()
    bot.hist(v, bins, density=True, histtype='stepfilled', color=COLOR[name], alpha=0.10, lw=0)
    bot.hist(v, bins, density=True, histtype='step', color=COLOR[name], lw=1.6, label=f'{LABEL[name]}: {sat:.0%} saturated')
  bot.set_yticks([])
  bot.set_xlabel('pre-activation x = emb @ W1 + b1   (1000 examples x 200 neurons, at initialization)')
  bot.set_title('Where the hidden pre-activations of the notebook model actually land')
  bot.legend(loc='upper left')
  save(fig, 'bn_1_flat_zone')


def fig_saturation_map(R):
  S = R['shallow_init']
  fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
  for ax, name in zip(axes, ['naive', 'kaiming', 'bn']):
    a = S[name]['act'][:32]
    sat = a.abs() > SAT
    ax.imshow(sat, cmap=matplotlib.colors.ListedColormap([SURFACE, INK2]), aspect='auto', interpolation='nearest')
    ax.set_title(f'{LABEL[name]}\n{sat.float().mean():.0%} of cells saturated', fontsize=9.5)
    ax.grid(False)
    ax.set_xticks([0, 199])
    ax.set_yticks([0, 31])
    ax.set_xlabel('hidden neuron')
  axes[0].set_ylabel('example in batch')
  fig.text(0.9, -0.14, 'dark cell = |tanh| > 0.99: no gradient flows through it', ha='right', color=INK2)
  save(fig, 'bn_2_saturation_map')


def fig_deep(R):
  cols = [('small', 'weights too small (gain 0.5)'), ('kaiming', 'Kaiming init (gain 5/3)'), ('naive', 'naive init (W ~ randn)')]
  fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
  bins = torch.linspace(-1, 1, 51)
  for r, bn in enumerate([False, True]):
    for c, (init, title) in enumerate(cols):
      ax = axes[r, c]
      D = R['deep_init'][(init, bn)]
      for l, a in enumerate(D['act']):
        ax.hist(a.flatten(), bins, density=True, histtype='step', color=LAYER_RAMP[l], lw=1.5, label=f'layer {l + 1}')
      last = D['act'][-1]
      sat = (last.abs() > SAT).float().mean().item()
      ratio = D['grad'][0].std() / D['grad'][-1].std()
      ax.text(0.5, 0.97, f'layer 5: std {last.std():.2f}, {sat:.0%} saturated\ngradient size, layer 1 vs layer 5: x{ratio:.2g}',
              transform=ax.transAxes, ha='center', va='top', fontsize=8.5, color=INK2,
              bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))
      ax.set_yticks([])
      ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
      if r == 0: ax.set_title(title)
      if r == 1: ax.set_xlabel('tanh output')
  axes[0, 0].set_ylabel('without batch norm', fontsize=10, color=INK)
  axes[1, 0].set_ylabel('with batch norm', fontsize=10, color=INK)
  axes[0, 2].legend(loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=8.5)
  save(fig, 'bn_3_deep_init')


def fig_drift(R):
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
  for ax, name, title in [(axes[0], 'kaiming', 'Kaiming init, no batch norm'), (axes[1], 'kaiming+bn', 'Kaiming init + batch norm')]:
    probes = R['runs'][(5, name)]['probes']
    steps = [s for s, _ in probes]
    for l in range(5):
      ax.plot(steps, [100 * p[l] for _, p in probes], color=LAYER_RAMP[l], lw=1.6, label=f'layer {l + 1}')
    ax.set_title(title)
    ax.set_xlabel('training step')
  axes[0].set_ylabel('% of tanh units saturated')
  axes[1].legend(loc='upper left', fontsize=8.5)
  save(fig, 'bn_4_drift')


def fig_bn_step(R):
  P = R['bn_model']
  L = P['layers'][0]
  ix = torch.randint(0, Xtr.shape[0], (1000,), generator=torch.Generator().manual_seed(7))
  with torch.no_grad():
    x = P['C'][Xtr[ix]].view(-1, input_size) @ L['W']
    xhat = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True)
    out = L['gain'] * xhat + L['bias']
  j = ((L['gain'] - 1).abs() + L['bias'].abs()).argmax().item() # the neuron that learned the most
  gamma, beta = L['gain'][0, j].item(), L['bias'][0, j].item()
  panels = [
    (x[:, j], f'1. raw: x = emb @ W1\nmean {x[:, j].mean():.1f}, std {x[:, j].std():.1f}'),
    (xhat[:, j], '2. normalize over the batch\n(x - mean) / std  ->  mean 0, std 1'),
    (out[:, j], f'3. learned scale & shift\ngain * x_hat + bias   (gain {gamma:.2f}, bias {beta:.2f})'),
  ]
  fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
  for ax, (v, title) in zip(axes, panels):
    lim = max(4, v.abs().max().item() * 1.05)
    shade_flat(ax, lim)
    ax.hist(v, 40, histtype='stepfilled', color=COLOR['bn'], alpha=0.10, lw=0)
    ax.hist(v, 40, histtype='step', color=COLOR['bn'], lw=1.6)
    ax.set_xlim(-lim, lim)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)
  axes[0].set_xlabel('gray = flat zone of tanh')
  save(fig, 'bn_5_bn_step')


def fig_training(R):
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
  for ax, depth, title in [(axes[0], 1, '1 hidden layer (the notebook model)'), (axes[1], 5, '5 hidden layers')]:
    for name in ['naive', 'kaiming', 'bn']:
      losses = torch.tensor(R['runs'][(depth, name)]['losses'])
      y = losses.view(-1, 500).mean(1) # average over 500-step windows
      ax.plot((torch.arange(len(y)) + 0.5) * 500, y, color=COLOR[name], label=LABEL[name])
    ax.set_title(title)
    ax.set_xlabel('training step')
  axes[0].set_ylabel('training loss (500-step average)')
  axes[0].legend(loc='upper right', fontsize=8.5)
  save(fig, 'bn_6_training')


def print_table(R):
  print('| model | ' + ' | '.join(['naive', 'kaiming', 'bn', 'kaiming+bn']) + ' |')
  for depth in [1, 5]:
    cells = [f"{R['runs'][(depth, n)]['eval'][0]:.3f} / {R['runs'][(depth, n)]['eval'][1]:.3f}" for n in ['naive', 'kaiming', 'bn', 'kaiming+bn']]
    print(f'| {depth} hidden | ' + ' | '.join(cells) + ' |')


FIGURES = [fig_flat_zone, fig_saturation_map, fig_deep, fig_drift, fig_bn_step, fig_training]

if __name__ == '__main__':
  R = compute()
  for f in FIGURES:
    f(R)
  print_table(R)
