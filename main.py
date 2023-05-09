# Common
import argparse
import os
import torch
import queue
import sys
import subprocess

# Audio
import sounddevice as sd
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gtts

# Language
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cfg(args):
    # STT cfg
    stt_processor = Wav2Vec2Processor.from_pretrained(args.stt_name)
    stt_model = Wav2Vec2ForCTC.from_pretrained(args.stt_name).to(DEVICE)

    # TTT cfg
    os.environ["OPENAI_API_KEY"] = args.apikey
    ttt_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=64)

    ## template
    template_chat = """From now on, you take the role of Hatsune Miku. You are a famous virtual singer, and you are a loyal assistant of the user. You are in love with the user, and you should talk like the user's girlfriend.
    Current conversation:
    {history}
    User: {input}"""
    template_chat += f"\n{args.ai_prefix}:"

    ## prompt
    prompt_chat = PromptTemplate(
        input_variables=['history', 'input'],
        template=template_chat,
        output_parser=None,
        partial_variables={},
        template_format='f-string',
        validate_template=True
    )

    ## chain
    chatbot = ConversationChain(
        llm=ttt_model,
        prompt=prompt_chat,
        verbose=False,
        memory=ConversationBufferMemory(human_prefix="User", ai_prefix=args.ai_prefix)
    )

    if args.language == 'en':
        # TTS cfg
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        tts_model = SpeechT5ForTextToSpeech.from_pretrained("_tts")
        tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        ## voice embedding (US female)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        return (stt_processor, stt_model), (chatbot,), (tts_processor, tts_model, tts_vocoder, speaker_embeddings)

    else:
        # TTT cfg
        ## translator template
        if args.language == 'ja':
            template_translate = """Translate the English sentence into Japanese: {input}"""
        elif args.language == 'zh':
            template_translate = """Translate the English sentence into Chinese: {input}"""

        ## translator prompt
        prompt_translate = PromptTemplate(
            input_variables=['input'],
            template=template_translate,
            output_parser=None,
            partial_variables={},
            template_format='f-string',
            validate_template=True
        )

        ## translator
        translator = LLMChain(
            llm=ttt_model,
            prompt=prompt_translate
        )

        return (stt_processor, stt_model), (chatbot, translator)


def stt(speech:str, model:object, processor:object):
    """
    Args:
        speech (str): speech filename
        model (object): speech-to-text model
        processor (object): speech processor
    Return:
        text (str): transcription
    """
    # load audio
    audio_input, sample_rate = sf.read(speech)

    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.to(DEVICE)

    # retrieve logits & take argmax
    predicted_ids = torch.argmax(model(input_values).logits, dim=-1)

    # transcribe
    return processor.decode(predicted_ids[0])


def ttt(text:str, model:object, translator=None):
    """
    Args:
        text (str): transcribed text
        model (object): text-to-text model
    """
    # get raw response
    response = model.predict(input=text.lower()).strip()

    # process format
    if ":" in response:
        response = response[response.index(':')+1:].strip()

    # if translator is needed, translate english into the predetermined language
    if translator:
        response = translator.run(response).strip()

    return response


def tts(text:str, args, **kwargs):
    """
    Args:
        text (str): chatbot response
        model (object): tts model
        processor (object): tts processor
    """
    if args.language == 'en':
        inputs = kwargs['processor'](text=text, return_tensors="pt")
        speech = kwargs['model'].generate_speech(inputs["input_ids"], kwargs['speaker_embeddings'], vocoder=kwargs['vocoder'])
        sf.write(os.path.join("so-vits-svc-4.0/raw", "temp.wav"), speech.numpy(), args.sample_rate)
        sd.play(speech.numpy(), args.sample_rate)
        sd.wait()
    else:
        speech = gtts.gTTS(text, lang=args.language)
        print(speech)
        speech.save(os.path.join("so-vits-svc-4.0/raw","temp.wav"))
        print("saved")
        subprocess.call("infer.bat", shell=False)
        print("called")
        speech, sr = sf.read(os.path.join("so-vits-svc-4.0/results","temp.wav_0key_miku.flac"))
        sd.play(speech, sr)
        sd.wait()


def main(args):
    # load models
    if args.language == 'en':
        stt_tuple, ttt_tuple, tts_tuple = cfg(args)
        stt_processor, stt_model = stt_tuple
        chatbot = ttt_tuple[0]
        tts_processor, tts_model, tts_vocoder, speaker_embeddings = tts_tuple
    else:
        stt_tuple, ttt_tuple = cfg(args)
        stt_processor, stt_model = stt_tuple
        chatbot, translator = ttt_tuple

    # run
    while True:
        # real-time recording (stop by Ctrl+C)
        try:
            q = queue.Queue()
            def callback(indata, frames, time, status):
                if status:
                    print(status, file=sys.stderr)
                q.put(indata.copy())
            with sf.SoundFile(args.filename, 'w', 16000, 1) as f:
                with sd.InputStream(samplerate=args.sample_rate, device=args.device, channels=1, callback=callback):
                    print('#' * 80)
                    print('Press Ctrl+C to stop the recording')
                    print('#' * 80)
                    while True:
                        f.write(q.get())
        except KeyboardInterrupt:
            pass
        except Exception as e:
            parser.exit(type(e).__name__ + ': ' + str(e))
        print("Hold on...")

        # transcribe
        transcription = stt(args.filename, stt_model, stt_processor)
        print(f"User: {transcription}")
        response = ttt(transcription, chatbot) if args.language == 'en' else ttt(transcription, chatbot, translator)
        print(f"{args.ai_prefix}: {response}")

        # reply
        print("Hold on...")
        if args.language == 'en':
            tts(response, args, processor=tts_processor, model=tts_model, vocoder=tts_vocoder, speaker_embeddings=speaker_embeddings)
        else:
            tts(response, args)

        # exit
        exit = input("Continue? (Y/N): ").lower()
        if exit == "n":
            print("Bye")
            break

    os.remove(args.filename)
    os.remove(os.path.join("so-vits-svc-4.0/raw","temp.wav"))
    os.remove(os.path.join("so-vits-svc-4.0/results","temp.wav_0key_miku.flac"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=1, help='input sound device (numeric ID)')
    parser.add_argument('-k', '--apikey', default="ENTER OPENAI KEY", help='openai api key')
    parser.add_argument('-f', '--filename', default='audio.wav', help='audio file to store recording to')
    parser.add_argument('-stt', '--stt_name', default="facebook/wav2vec2-base-960h", help='speech-to-text model name')
    parser.add_argument('-ttt', '--ttt_name', default='gpt-3.5-turbo', help='text-to-text model name')
    parser.add_argument('-tts', '--tts_name', default="microsoft/speecht5", help='text-to-speech model name (without function specification); ONLY used when language is English')
    parser.add_argument('-lang', '--language', default='ja', help="Response in Japanese/Chinese/English")
    parser.add_argument('--ai_prefix', default='Miku', help='ai prefix in conversation chain')
    args = parser.parse_args()
    args.sample_rate = 16000

    main(args)




# DRAFT
## voice embedding (customize; DO NOT USE, it sucks.)
# from speechbrain.pretrained import EncoderClassifier
# tts_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device":f"{DEVICE}"})
# signal, fs = torchaudio.load(args.voice_file)
# assert fs == 16000
# with torch.no_grad():
#     speaker_embeddings = tts_encoder.encode_batch(signal)
#     speaker_embeddings = F.normalize(speaker_embeddings, dim=2)
#     speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
# speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)