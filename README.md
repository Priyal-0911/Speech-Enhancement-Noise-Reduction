# Speech Enhancement / Noise Reduction

This repository demonstrates the process of enhancing and separating mixed audio sources, such as isolating speech from background noise, using a pre-trained model. Given a mixed audio file, the model separates and enhances individual audio sources, allowing you to save the enhanced output for further use.

![Speech Enhancement](https://github.com/Priyal-0911/Speech-Enhancement-Noise-Reduction/blob/32e75ac9078879e0db1e829a6d59ac01fd98cfb0/display.avif)

## Requirements

- Python 3.6+
- `torch`
- `torchaudio`
- `speechbrain`
- `IPython`

To install the necessary dependencies, use the following:

```bash
pip install torch torchaudio speechbrain ipython
```

## Setup

1. Clone the Repository

```bash
git clone https://github.com/Priyal-0911/Speech-Enhancement-Noise-Reduction.git
```

2. Download Pretrained Model
   The script uses the pretrained Sepformer model to separate audio sources. The model weights will be automatically downloaded from SpeechBrain's repository the first time you run the script.

3. Run the Code
   Run the Python script to separate audio sources from a given .wav file. The script will read the input audio, process it using the Sepformer model, and save the separated audio as an output file.

```python
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio, torch
from IPython.display import Audio

# Load pretrained Sepformer model
model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')

# Separate audio sources from file
audio_sources = model.separate_file(path='/path/to/your/file.wav')

# Save the separated audio to a file
torchaudio.save("output.mp3", audio_sources[:, :, 0], 16000)
```

Replace /path/to/your/file.wav with the path to the input audio file you want to process.

4. Check Output
   The script saves the output audio as output.mp3. You can open it with any media player.

## File Structure

```bash
├── README.md
├── main.py
├── pretrained_models/     # It will automatically create when you run the script
└── output.mp3
```

## Model

The model used in this repository is trained on the WHAM dataset for speech enhancement. The model automatically separates different sources in the input file.

## Troubleshooting

Audio Output Not Playing: If the output audio is not playing correctly, ensure the file format is supported by your audio player. Convert it to a more common format like .wav if necessary.

## Contributing

If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.
