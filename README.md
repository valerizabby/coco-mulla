## Description

This fork was made to compare coco-mulla model with ordinary musicgen. To do so, several steps was made, including:
1. Prepare a test set;
2. Run in inference coco-mulla and music gen;
3. Compare results.

Here you can see a step-by-step guide to reproduce my results.

- OS: MacOs & Google Colab
- Python: 3.11

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
Now we should run coco-mulla on inference on the same tracks, but there is an issue with compatability of coco-mulla with MacOs.
So I decided to use A100 GPU on Google Colab. The ipynb is called `coco_mulla_inference.ipynb` and its prepared to run in Google Colab.

## Results

### Metrics

```
musicgen_pitch_corr             -0.069034
musicgen_energy_corr             0.143787
musicgen_energy_ratio            2.319782
chord-only_pitch_corr            0.484476
chord-only_energy_corr           0.040173
chord-only_energy_ratio          2.036327
chord-drums_pitch_corr           0.266954
chord-drums_energy_corr          0.415343
chord-drums_energy_ratio         1.764437
chord-midi_pitch_corr            0.455966
chord-midi_energy_corr          -0.000249
chord-midi_energy_ratio          2.175005
chord-drums-midi_pitch_corr      0.590524
chord-drums-midi_energy_corr     0.333540
chord-drums-midi_energy_ratio    1.836883
```

### Analysis 


- The use of structural features (chords, midi, drums) makes CoCoMulla a much more consistent model with the original in all key musical aspects. 
- MusicGen is a powerful, but unstructured bassline.

## Resources

- **Lakh MIDI Dataset (LMD)**  
  A collection of thousands of MIDI files aligned with real audio.  
  📎 [https://colinraffel.com/projects/lmd/](https://colinraffel.com/projects/lmd/)

- **Million Song Dataset (MSD)**  
  Metadata for the Lakh dataset with artist, title, and genre info.  
  📎 [http://millionsongdataset.com](http://millionsongdataset.com)

- **Extracted Chords for LMD**  
  Annotated chord sequences for the Lakh dataset, hosted on Hugging Face.  
  📎 [LMD Chords @ HuggingFace](https://huggingface.co/datasets/ohollo/lmd_chords/blob/4d6815cdd528bd1e99dcdefcb06d6f40429ec128/README.md)

- **Fixed Audiocraft (Mac-compatible fork)**  
  A fork of the Meta Audiocraft repository with Mac and local compatibility fixes.  
  📎 [https://github.com/GrandaddyShmax/audiocraft_plus](https://github.com/GrandaddyShmax/audiocraft_plus)

- **MusicGen Small (official model weights)**  
  Lightweight version of MusicGen for local inference and evaluation.  
  📎 [https://huggingface.co/facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small)