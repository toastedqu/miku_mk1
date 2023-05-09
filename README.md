# Miku mk.1
Given the rise of availability of generative AI models, this project is my first naive attempt to create an actual "chat"bot that chats with you through speech rather than text.

The full functionality consists of 4 parts:

## STT (Speech-To-Text)
[Wav2Vec2](https://arxiv.org/pdf/2006.11477.pdf) from Facebook AI was used as the STT model. It is imperfect but still the best among all I've tested at the moment. I might add customization to model choice later.

## TTT (Text-To-Text)
[ChatGPT](https://openai.com/blog/chatgpt) from OpenAI was used as the chatbot engine. I might also add customization to LLM choice later. It executes a 2-step process: 1) Response Generation 2) Machine Translation.

## TTS (Text-To-Speech)
[SpeechT5](https://arxiv.org/pdf/2110.07205.pdf) from Microsoft was used as the TTS model for English responses. My current voice converter was trained on Japanese and thus does not work well on English speech, so the file sticks to the SLT (US Female) speaker embedding from the [CMU ARCTIC](http://www.festvox.org/cmu_arctic/) dataset.

## VC (Voice Conversion)
This is the most complex part of the entire project. To get the voice of a specific speaker, you need to fine-tune the [SoftVC VITS SVC](https://github.com/svc-develop-team/so-vits-svc) models onto a self-prepared audio dataset. 

### Train
1. Collect at least 2 hours of audio data from the speaker alone. (optional: less background noise, 44.1kHz sample rate, 16 bit, .wav format)
2. Use [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) to remove as much unnecessary noises/echos as possible so that your voice is the only sound that exist in the files. You have 3 options. It is preferable to do all 3 of them to get max cleansing effect, but it might be an overkill for certain audio files. Tick "Vocals Only" for all of them. Tick "GPU Conversion" if GPU is available.
    - Select [DEMUCS](https://arxiv.org/pdf/1909.01174.pdf) as "process method" with an arbitrary version.
    - Select [MDX-Net](https://arxiv.org/pdf/2111.12203.pdf) as "process method and "UVR-MDX-NET Main" as the model version.
    - Select VR Architecture as "process method and "HP-Karaoke-UVR" as the model version.
3. Use [Audio Slicer](https://github.com/openvpi/audio-slicer) to slice your audio files into segments of 5-15s. 
4. Follow the [SoftVC VITS SVC](https://github.com/svc-develop-team/so-vits-svc) guideline step-by-step to train and test your own speaker converter.

The audio quality has a strong impact on the final model performance. If large data is not available, it is preferable to have a small set of high-quality audio data of the speaker voice alone than a large set of med-quality audio data. I tested both, and the latter failed miserably.

Despite having many voice removing options, they cannot remove all noises, specifically the inseperable ones entangled with the speaker voice. It is better to have high-quality recordings from the start rather than relying on UVR5.

### Infer
The TTS inference includes two parts: 1) Google TTS 2) VC. SpeechT5 is not reliable for languages outside English, and Google TTS offers higher audio quality. 

Please edit "infer.bat" and other variables to fit your own needs.

## Remark
Pros: it works

Cons: high latency; only interactable from terminal right now.

Due to the mismatch in pytorch and cuda versions between the voice converter and main.py, it is currently difficult to discard the batch script and merge the voice converter into main.py. If this is doable latency can be eliminated.