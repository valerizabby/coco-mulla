import os
import zipfile
import argparse
import librosa
import torch
import torch.nn.functional as F
import numpy as np

from coco_mulla.models import CoCoMulla
from coco_mulla.utilities import get_device, np2torch, mkdir, read_lst
from coco_mulla.utilities.encodec_utils import extract_rvq, save_rvq
from coco_mulla.utilities.symbolic_utils import process_midi, process_chord
from coco_mulla.utilities.sep_utils import separate
from config import TrainCfg

device = get_device()

def crop(x, mode, sample_sec, res, offset=0):
    xlen = x.shape[1] if mode in ["chord", "midi"] else x.shape[-1]
    sample_len = int(sample_sec * res) + 1
    if xlen < sample_len:
        if mode in ["chord", "midi"]:
            x = np.pad(x, ((0, 0), (0, sample_len - xlen), (0, 0)))
        else:
            x = F.pad(x, (0, sample_len - xlen), "constant", 0)
        return x
    st = offset * res
    ed = int((offset + sample_sec) * res) + 1
    if mode in ["chord", "midi"]:
        return x[:, st: ed]
    return x[:, :, st: ed]

def load_data(audio_path, chord_path, midi_path, offset):
    sr = TrainCfg.sample_rate
    res = TrainCfg.frame_res
    sample_sec = TrainCfg.sample_sec

    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav = np2torch(wav).to(device)[None, None, ...]
    wavs = separate(wav, sr)
    drums_rvq = extract_rvq(wavs["drums"], sr=sr)

    chord, _ = process_chord(chord_path)
    midi, _ = process_midi(midi_path)

    chord = crop(chord[None, ...], "chord", sample_sec, res)
    pad_chord = chord.sum(-1, keepdims=True) == 0
    chord = np.concatenate([chord, pad_chord], -1)

    midi = crop(midi[None, ...], "midi", sample_sec, res, offset=offset)
    drums_rvq = crop(drums_rvq[None, ...], "drums_rvq", sample_sec, res, offset=offset)

    chord = torch.from_numpy(chord).to(device).float()
    midi = torch.from_numpy(midi).to(device).float()
    drums_rvq = drums_rvq.to(device).long()

    return drums_rvq, midi, chord

def generate_mask(xlen):
    names = ["chord-only"]
    mask = torch.zeros([1, 2, xlen]).to(device)
    mask[0, 0] = 1  # подаем только аккорды
    return mask, names

def wrap_batch(drums_rvq, midi, chord, cond_mask, prompt):
    num_samples = len(cond_mask)
    midi = midi.repeat(num_samples, 1, 1)
    chord = chord.repeat(num_samples, 1, 1)
    drums_rvq = drums_rvq.repeat(num_samples, 1, 1)
    prompt = [prompt] * num_samples
    return {
        "seq": None,
        "desc": prompt,
        "chords": chord,
        "num_samples": num_samples,
        "cond_mask": cond_mask,
        "drums": drums_rvq,
        "piano_roll": midi,
        "mode": "inference",
    }

def inference_from_zip(zip_path, model_path, num_layers, latent_dim, onset=0):
    extract_dir = "unzipped_data"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    model = CoCoMulla(TrainCfg.sample_sec, num_layers=num_layers, latent_dim=latent_dim).to(device)
    model.load_weights(model_path)
    model.eval()

    for folder in sorted(os.listdir(extract_dir)):
        path = os.path.join(extract_dir, folder)
        if not os.path.isdir(path):
            continue

        print(f"🎧 Generating for {folder}...")

        audio_path = os.path.join(path, "audio.wav")
        midi_path = os.path.join(path, "midi.mid")
        chord_path = os.path.join(path, "chords.txt")
        prompt_path = os.path.join(path, "prompt.txt")

        drums_rvq, midi, chord = load_data(audio_path, chord_path, midi_path, offset=onset)
        cond_mask, names = generate_mask(drums_rvq.shape[-1])
        with open(prompt_path) as f:
            prompt = f.read().strip()

        batch = wrap_batch(drums_rvq, midi, chord, cond_mask, prompt)
        with torch.no_grad():
            pred = model(**batch)

        output_path = os.path.join(path, "cocomulla_output")
        mkdir(output_path)
        save_rvq(
            output_list=[os.path.join(output_path, name) for name in names],
            tokens=pred
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--onset", type=int, default=0)
    args = parser.parse_args()

    inference_from_zip(
        zip_path=args.zip_path,
        model_path=args.model_path,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        onset=args.onset
    )