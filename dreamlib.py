"""
Libreria DeepDream: il codice che puoi modificare direttamente su GitHub.

Dopo ogni modifica, in Colab riesegui la cella di setup (fa git pull)
e poi la cella Genera: non serve toccare il notebook.
"""

import random

import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision.transforms.functional import to_tensor

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Strato intermedio di partenza per ogni rete (usato quando strato = "auto")
STRATI_DEFAULT = {
    "vgg19": "features.20",
    "vgg16": "features.20",
    "alexnet": "features.9",
    "googlenet": "inception4c",
    "resnet50": "layer3",
}


class StopForward(Exception):
    # Interrompe il forward appena letto lo strato scelto
    pass


class Dreamer:
    def __init__(self, model_name="vgg19"):
        self.name = model_name
        self.model = getattr(models, model_name)(weights="DEFAULT").eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)   # i pesi restano fermi: cambia solo l'immagine
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.handle = None
        self.store = {}
        self.channel = None

    def list_layers(self):
        for name, m in self.model.named_modules():
            if name and name.count(".") <= 1:
                print(f"{name:30s} {m.__class__.__name__}")

    def set_layer(self, layer, channel=None):
        if self.handle is not None:
            self.handle.remove()
        modules = dict(self.model.named_modules())
        if layer not in modules:
            raise ValueError(f"Strato '{layer}' non trovato: usa d.list_layers()")

        def hook(_module, _inp, out):
            self.store["act"] = out
            raise StopForward

        self.handle = modules[layer].register_forward_hook(hook)
        self.channel = channel

    def objective(self, x):
        # L(x) = 1/2 * mean(a^2), su tutto lo strato o su un solo canale.
        # Per provare altri obiettivi, e' qui che si interviene.
        try:
            self.model((x - self.mean) / self.std)
        except StopForward:
            pass
        a = self.store["act"]
        if self.channel is not None:
            a = a[:, self.channel]
        return 0.5 * (a ** 2).mean()


def resolve_layer(d, modello, strato):
    # "auto" -> strato di default; se lo strato non esiste nella rete, ripiega sul default
    usato = STRATI_DEFAULT[modello] if strato == "auto" else strato
    if usato not in dict(d.model.named_modules()):
        print(f"Strato '{usato}' non esiste in {modello}: uso {STRATI_DEFAULT[modello]}")
        usato = STRATI_DEFAULT[modello]
    return usato


def symmetrize(x, mode):
    # Forza la simmetria (serve un'immagine quadrata per la rotazione)
    if mode in ("speculare", "entrambe"):
        x = (x + torch.flip(x, [3])) / 2
    if mode in ("rotazionale", "entrambe"):
        x = sum(torch.rot90(x, k, (2, 3)) for k in range(4)) / 4
    return x


def dream(d, x, steps, lr, jitter, symmetry):
    # Gradient ascent sui pixel: x <- x + eta * g / mean|g|
    for _ in range(steps):
        ox = random.randint(-jitter, jitter) if jitter else 0
        oy = random.randint(-jitter, jitter) if jitter else 0
        xs = torch.roll(x, shifts=(ox, oy), dims=(2, 3)).detach().requires_grad_(True)
        (g,) = torch.autograd.grad(d.objective(xs), xs)
        g = torch.roll(g, shifts=(-ox, -oy), dims=(2, 3))
        g = g / (g.abs().mean() + 1e-8)
        x = (x + lr * g).clamp(0, 1)
        x = symmetrize(x, symmetry).detach()
    return x


def dream_octaves(d, base, octaves, scale, steps, lr, jitter, symmetry):
    H, W = base.shape[2:]
    sizes = [(int(H / scale ** i), int(W / scale ** i)) for i in reversed(range(octaves))]
    detail = None
    for k, (h, w) in enumerate(sizes, start=1):
        img_o = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
        if detail is None:
            x = img_o
        else:
            x = img_o + F.interpolate(detail, size=(h, w), mode="bilinear", align_corners=False)
        x = dream(d, x.clamp(0, 1), steps, lr, jitter, symmetry)
        detail = x - img_o
    return x


def make_start(start, size, image_path=None, square=False):
    if start == "rumore":
        x = 0.5 + 0.1 * torch.randn(1, 3, size, size)
    else:
        img = Image.open(image_path).convert("RGB")
        if square:   # ritaglio quadrato centrale
            w, h = img.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
        img.thumbnail((size, size))
        x = to_tensor(img).unsqueeze(0)
    return x.clamp(0, 1).to(device)
