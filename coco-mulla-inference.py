import argparse
import librosa
import os
import torch
torch.cuda.set_per_process_memory_fraction(0.9, 0)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch.nn.functional as F
import numpy as np

from coco_mulla.models import CoCoMulla
from coco_mulla.utilities import get_device, np2torch, mkdir, read_lst
from coco_mulla.utilities.encodec_utils import extract_rvq, save_rvq
from coco_mulla.utilities.symbolic_utils import process_midi, process_chord
from coco_mulla.utilities.sep_utils import separate
from config import TrainCfg

device = get_device()

from pydub import AudioSegment
import tempfile

def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

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

    if audio_path.endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)

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

import gc
import torch

def inference_from_folder(data_dir, model_path, num_layers, latent_dim, onset=0):
    torch.cuda.empty_cache()
    gc.collect()
    model = CoCoMulla(TrainCfg.sample_sec, num_layers=num_layers, latent_dim=latent_dim).to(device)
    model.load_weights(model_path)
    model.eval()

    for folder in sorted(os.listdir(data_dir)):
        torch.cuda.empty_cache()
        gc.collect()
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            continue

        print(f"🎧 Generating for {folder}...")

        audio_dir = os.path.join(path, "audio")
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]
        if not audio_files:
            raise FileNotFoundError(f"No .mp3 file found in {audio_dir}")
        audio_path = os.path.join(audio_dir, audio_files[0])

        midi_dir = os.path.join(path, "midi")
        midi_files = [f for f in os.listdir(midi_dir) if f.endswith(".mid")]
        if not midi_files:
            raise FileNotFoundError(f"No .mid file found in {midi_dir}")
        midi_path = os.path.join(midi_dir, midi_files[0])

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
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--onset", type=int, default=0)
    args = parser.parse_args()

    inference_from_folder(
        data_dir=args.data_dir,
        model_path=args.model_path,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        onset=args.onset
    )