import os
import sys
import subprocess
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import json
from collections import namedtuple
from pathlib import Path


CACHE = Path("./cache")

if not os.path.exists(CACHE):
    os.makedirs(CACHE, exist_ok=True)


def do_stemming(file, stemming):
    out_file = CACHE / "htdemucs" / os.path.splitext(os.path.basename(file))[0] / "vocals.wav"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        if os.path.getsize(out_file) == 0:
            return file
        else:
            return out_file

    if stemming:
        print("Splitting vocals")
        proc = subprocess.Popen(
            f'"{sys.executable}" -m demucs.separate -n htdemucs --two-stems=vocals "{file}" -o "temp_outputs"',
            shell=False
        )
        proc.communicate()

        if proc.returncode != 0:
            logging.warning(
                "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
            )
            # create empty file
            open(out_file, "w+").close()
            return file
        else:
            return out_file
    else:
        print("Using base audio file")
        return file


if __name__ == '__main__':
    mtypes = {'cpu': 'int8', 'cuda': 'float16'}
    device = "cpu"
    language = "nl"
    file = "./verg/20230629 raad trim.mp3"
    file = os.path.abspath(file)
    assert os.path.exists(file)
    model_name = "large-v2"
    stemming = False

    vocal_target = do_stemming(file, stemming)

    # Run selected model
    whisper_model = WhisperModel(
        model_name, device=device, compute_type=mtypes[device]
    )

    segments, info = whisper_model.transcribe(
        vocal_target, beam_size=1, word_timestamps=True, language=language
    )
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    whisper_results = []
    with open("vergardering.txt", "w+", encoding="utf-8", errors="ignore") as f:
        for segment in segments:
            print(segment.text)
            f.write(segment.text)
            f.flush()
            whisper_results.append(segment._asdict())
    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()

    print("Aligning results")
    if info.language in wav2vec2_langs:
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, device
        )
        word_timestamps = result_aligned["word_segments"]
        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})


    # convert audio to mono for NeMo combatibility
    print("Converting audio to mono")
    signal, sample_rate = librosa.load(vocal_target, sr=None)
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    soundfile.write(os.path.join(temp_path, "mono_file.wav"), signal, sample_rate, "PCM_24")

    # Initialize NeMo MSDD diarization model
    print("Initialize diarization")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping


    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            print(word)
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        logging.warning(
            f'Punctuation restoration is not available for {info.language} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    with open(f"{os.path.splitext(file)[0]}.txt", "w", encoding="utf-8-sig", errors="ignore") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(file)[0]}.srt", "w", encoding="utf-8-sig", errors="ignore") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
