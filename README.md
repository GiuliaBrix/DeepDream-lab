# DeepDream-lab

A small, hackable implementation of **DeepDream** that works with any convolutional network from `torchvision` (VGG, GoogLeNet, ResNet, AlexNet...). It includes symmetry options for mandala-like patterns and a recursive zoom loop for fractal-style animations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GiuliaBrix/DeepDream-lab/blob/main/deepdream_notebook_v3.ipynb)

## Try it

1. Click the **Open In Colab** button above (a Google account is required).
2. In Colab: **Runtime → Change runtime type → GPU**.
3. Run the cells in order: **Setup**, **Parameters**, **Generate**.
4. Change the fields in **Parameters**, run that cell again, then run **Generate** again.

To start from your own photo, upload it in the Colab file panel (main folder, not inside `DeepDream-lab`), set `partenza` to `immagine` and write the file name in `immagine`.

## Parameters

| Parameter | What it does |
|---|---|
| `modello` | Network used (`vgg19`, `vgg16`, `googlenet`, `resnet50`, `alexnet`) |
| `strato` | Layer to amplify. `auto` picks a mid-level layer for the chosen model |
| `canale` | Single channel of the layer (`-1` = the whole layer) |
| `partenza` | Start from random noise (`rumore`) or from your photo (`immagine`) |
| `immagine` | File name of your photo |
| `simmetria` | `nessuna`, `speculare` (mirror), `rotazionale` (4-fold rotation), `entrambe` |
| `dimensione` | Image side in pixels |
| `ottave`, `scala_ottave` | Number of octaves and scale factor between them |
| `passi` | Gradient ascent steps per octave |
| `lr` | Step size |
| `jitter` | Random shift in pixels at each step |

Lower layers produce textures and repeating patterns, higher layers produce complex shapes (eyes, animals). Layer names depend on the model: `features.N` for VGG, `inceptionXY` for GoogLeNet, `layerN` for ResNet.

## How it works

The network weights stay frozen. At each step the image is passed through the network up to a chosen layer, the objective `L(x) = 1/2 * mean(a²)` is computed on that layer's activations, and the **pixels** are updated by normalized gradient ascent: `x ← x + η · g / mean|g|`. This is repeated over several octaves (image scales), reinserting the generated detail at each scale.

## Repository structure

```
dreamlib.py                    # the actual code (objective, octaves, symmetry...)
deepdream_notebook_v3.ipynb    # Colab notebook that downloads and uses dreamlib.py
```

The notebook clones this repository at every session, so any change to `dreamlib.py` is picked up by re-running the **Setup** cell.

## Ideas to experiment with

- Edit `Dreamer.objective` in `dreamlib.py` to change what gets amplified.
- Explore different channels of the same layer: each one has a very different character.
- Try different networks: GoogLeNet and VGG give very different looks.

## Notes

- Pretrained weights are downloaded automatically by `torchvision` the first time a model is used.
- On CPU it works but is slow: reduce `dimensione`, `ottave` and `passi`.
