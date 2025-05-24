## Description

This fork was made to compare coco-mulla model with ordinary musicgen. To do so, several steps was made, including:
1. Prepare a test set;
2. Run in inference coco-mulla and music gen;
3. Compare results.

Here you can see a step-by-step guide to reproduce my results.

OS: MacOs
Python: 3.11

### 1. Prepare a test set.

First of all, we should gather and combine data, to make as it clear as possible. I will use the [LMD dataset](https://colinraffel.com/projects/lmd/), specifically LMD-aligned. 

P.S. Getting a full version requires asking for it: craffel@gmail.com

Secondly, we should get a metadata for this dataset, for example from [hugging face](https://huggingface.co/datasets/ohollo/lmd_chords/blob/4d6815cdd528bd1e99dcdefcb06d6f40429ec128/README.md).

Lastly, lets overview a preparing job:
1. Get midi trasks from `lmd_aligned`
2. Get audio tracks from `lmd_aligned_mp3`
3. Structure `testset` directory: 
```
testset/
├── track_001/
│   ├── audio/
│   │   ├── TRAAAGR128F425B14B.mp3
│   ├── midi/
│   │   ├── ...
```
4. Run an `extractor.py`, which takes metadata from LMD and puts it into `testset`.
5. Ask a GPT-4o to generate a promts for music generation based on a metadata of tracks. We got:
```
An energetic electronic pop song by Cyndi Lauper titled "Into The Nightlife".
A driving trance remix by Rene Ablaze titled "Metamorpic".
A symphonic metal track by Nightwish with powerful female vocals.
A Brazilian MPB (Música Popular Brasileira) classic performed by Elis Regina.
A heavy glam rock anthem by Alice Cooper titled "Poison".
A deep progressive house track by Leftfield with hypnotic groove.
A romantic Latin ballad by Julio Iglesias titled "Manuela".
An emotional French pop ballad by Lara Fabian.
A 60s sunshine pop tune by The Association with rich harmonies.
A heartfelt French chanson by Jean-Jacques Goldman.
```
### 2. Inference
#### MusicGen

To use MusicGen on MacOs specific steps required. `audiocraft` library expect preinstalled `xformermer`, which is not suitable with CPU. So I will use a fork:
```link
https://github.com/cocktailpeanutlabs/audiocraft_plus/tree/mac-os-fix
```

#### Installation:

1. 
```bash 
pip install git+https://github.com/cocktailpeanutlabs/audiocraft_plus.git@mac-os-fix 
```
2. Import model as usual, but on CPU
```python
model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
```

#### Inference:

Inference code stored in `musicgen_inference.py`.

#### Coco-mulla 
Now we should run coco-mulla on inference on the same tracks, but there is an issue with compatability of coco-mulla with MacOs. General idea: prepare patch to adapt is on Mac (same as with audiocraft-plus).


[//]: # (I want to use `/facebook/musicgen-small`, but it is not possible without preinstalled `xformers`, which is not suitable for MacOs.)

[//]: # (I decided to patch a fix into library to go through this problem.)

[//]: # ()
[//]: # (#### Patch)

[//]: # (1. Open `venv/lib/python3.11/site-packages/audiocraft/modules/transformer.py`)

[//]: # (2. Comment all `xformers` imports.)

[//]: # (3. Change )

[//]: # (```python)

[//]: # (    from xformers.ops import LowerTriangularMask)

[//]: # (```)

[//]: # (on the raw implementation:)

[//]: # (```python )

[//]: # (    def dummy_LowerTriangularMask&#40;size&#41;:)

[//]: # (        # fallback for CPU)

[//]: # (        return torch.tril&#40;torch.ones&#40;size, size&#41;&#41;.to&#40;torch.bool&#41;)

[//]: # (    )
[//]: # (    LowerTriangularMask = dummy_LowerTriangularMask)

[//]: # (```)

[//]: # (   )
## Links

LMD dataset
https://colinraffel.com/projects/lmd/

MSD dataset (lakh's metadata)
http://millionsongdataset.com

Extracted Chords
https://huggingface.co/datasets/ohollo/lmd_chords/blob/4d6815cdd528bd1e99dcdefcb06d6f40429ec128/README.md

Fixed Audiocraft
https://github.com/GrandaddyShmax/audiocraft_plus

Musicgen small
https://huggingface.co/facebook/musicgen-small


# TRY THIS 
how to install audiocraft
https://huggingface.co/undefinedxyz/musicgen