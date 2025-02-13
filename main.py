from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio, torch
from IPython.display import Audio
model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
audio_sources = model.separate_file(path='/home/ml/Music/jackhammer.wav')

torchaudio.save("output.mp3", audio_sources[:, :, 0], 16000)
