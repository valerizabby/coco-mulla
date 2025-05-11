from audiocraft.models import MusicGen
import torchaudio
import os

model = MusicGen.get_pretrained('small')
model.set_generation_params(duration=10)

root_dir = "testset"

for folder in sorted(os.listdir(root_dir)):
    path = os.path.join(root_dir, folder)
    if not os.path.isdir(path):
        continue

    with open(os.path.join(path, "prompt.txt")) as f:
        prompt = f.read().strip()

    print(f"🎧 Generating for {folder}...")
    wav = model.generate([prompt])[0]
    output_path = os.path.join(path, "musicgen_output.wav")
    torchaudio.save(output_path, wav.detach().cpu(), 32000)