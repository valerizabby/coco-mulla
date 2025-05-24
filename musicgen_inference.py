from audiocraft.models import MusicGen
import torchaudio
import os

model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
model.set_generation_params(duration=5)

root_dir = "testset"

for folder in sorted(os.listdir(root_dir)):
    path = os.path.join(root_dir, folder)
    if not os.path.isdir(path):
        continue

    with open(os.path.join(path, "prompt.txt")) as f:
        prompt = f.read().strip()

    print(f"Generating for {folder}...")
    output_path = os.path.join(path, "musicgen_output.wav")

    wav = model.generate([prompt])
    wav = wav[0].cpu()
    torchaudio.save(output_path, wav, 32000)