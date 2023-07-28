import os
import sys
import subprocess
from helpers import *
import torch
import re
import logging
import json
from pathlib import Path


CACHE = Path("./cache")

if not os.path.exists(CACHE):
    os.makedirs(CACHE, exist_ok=True)


def do_stemming(file, stemming, verbose=False):
    out_file = CACHE / os.path.splitext(os.path.basename(file))[0] / "htdemucs" / "vocals.wav"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        if os.path.getsize(out_file) == 0:
            return file
        else:
            return out_file

    if stemming:
        if verbose:
            print("Splitting vocals")
        proc = subprocess.Popen(
            f'"{sys.executable}" -m demucs.separate -n htdemucs --two-stems=vocals "{file}" -o "{os.path.abspath(out_file)}"',
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
        if verbose:
            print("Using base audio file")
        return file


def conv_segment(segment):
    asdict = segment._asdict()
    asdict["words"] = [word._asdict() for word in asdict["words"]]
    return asdict


def transcribe(file, vocal_target, model_name, device=None, compute_type=None, language=None, raw_output=None, verbose=False):
    out_folder = CACHE / os.path.splitext(os.path.basename(file))[0] / "whisper"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    else:
        if os.path.exists(out_folder / "info.json") and os.path.exists(out_folder / "segments.json"):
            if verbose:
                print("Found cached transcription")
            with open(out_folder / "segments.json", "r") as f:
                segments = json.load(f)
            with open(out_folder / "info.json", "r") as f:
                from faster_whisper.transcribe import TranscriptionInfo
                info = TranscriptionInfo(*json.load(f))
            return segments, info

    from faster_whisper import WhisperModel
    # Run selected model
    whisper_model = WhisperModel(
        model_name, device=device, compute_type=compute_type
    )

    segments, info = whisper_model.transcribe(
        vocal_target, beam_size=1, word_timestamps=True, language=language
    )
    if verbose:
        print(f"Detected language '{info.language}' with probability {info.language_probability}")

    whisper_results = []
    if raw_output is not None:
        with open(raw_output, "w+", encoding="utf-8", errors="ignore") as f:
            for segment in segments:
                if verbose:
                    print(segment.start, segment.text)
                f.write(segment.text + "\n")
                f.flush()
                whisper_results.append(conv_segment(segment))
    else:
        for segment in segments:
            if verbose:
                print(segment.start, segment.text)
            whisper_results.append(conv_segment(segment))

    with open(out_folder / "segments.json", "w+") as f:
        json.dump(whisper_results, f, indent=2)
    with open(out_folder / "info.json", "w+") as f:
        json.dump(info, f, indent=2)

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    return whisper_results, info


def align_results(file, vocal_target, whisper_results, language, device, verbose=False):
    out_file = CACHE / os.path.splitext(os.path.basename(file))[0] / "whisperx" / "aligned.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        if verbose:
            print("Found cached alignment")
        with open(out_file, "r") as f:
            return json.load(f)

    import whisperx

    if language in wav2vec2_langs:
        if verbose:
            print("Loading alignment model")
        alignment_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        if verbose:
            print("Aligning text")

        # custom iterator for verbose logging of progress
        class _LogIterResults:
            def __init__(self):
                self.idx = 0

            def __iter__(self):
                for i, segment in enumerate(whisper_results):
                    if verbose:
                        print(f"Aligning segment {i} / {len(whisper_results)} @{segment['start']}-{segment['end']}")
                    yield segment

            def __getitem__(self, item):
                return whisper_results[item]

            def __setitem__(self, key, value):
                whisper_results[key] = value

        result_aligned = whisperx.align(
            _LogIterResults(), alignment_model, metadata, vocal_target, device
        )
        word_timestamps = result_aligned["word_segments"]
        if verbose:
            print("Clearing cache")
        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"text": word["word"], "start": word["start"], "end": word["end"]})

    with open(out_file, "w+") as f:
        json.dump(word_timestamps, f, indent=2)

    return word_timestamps


def diarize_speakers(file, vocal_target, verbose=False):
    # convert audio to mono for NeMo combatibility
    if verbose:
        print("Converting audio to mono")
    temp_path = CACHE / os.path.splitext(os.path.basename(file))[0]
    if not os.path.exists(temp_path / "nemo" / "pred_rttms" / "mono_file.rttm"):
        if not os.path.exists(temp_path / "mono_file.wav"):
            import librosa
            import soundfile
            signal, sample_rate = librosa.load(vocal_target, sr=None)
            os.makedirs(temp_path, exist_ok=True)
            soundfile.write(temp_path / "mono_file.wav", signal, sample_rate, "PCM_24")

        # Initialize NeMo MSDD diarization model
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        if verbose:
            print("Initialize diarization")
        msdd_model = NeuralDiarizer(cfg=create_config(
            temp_path / "nemo",
            temp_path / "mono_file.wav",
            # temp_path / "nemo" / (os.path.splitext(os.path.basename(file))[0] + ".rttm"),
            # temp_path / "nemo" / (os.path.splitext(os.path.basename(file))[0] + ".uem")
        )).to(device)
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()
    elif verbose:
        print("Found output rttm file, using it")

    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    with open(temp_path / "nemo" / "pred_rttms" / "mono_file.rttm", "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        from deepmultilingualpunctuation import PunctuationModel
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labeled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts)
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

    return wsm, ssm


if __name__ == '__main__':
    mtypes = {'cpu': 'int8', 'cuda': 'float16'}
    device = "cpu"
    language = "nl"
    file = "./verg/20230629 raad trim.mp3"
    file = os.path.abspath(file)
    assert os.path.exists(file)
    model_name = "large-v2"
    # model_name = "nlmodel"
    stemming = False

    print("Preparing vocal target")
    vocal_target = do_stemming(file, stemming)
    print("Transcribing audio file")
    whisper_results, info = transcribe(file, vocal_target, model_name, device, mtypes[device], language=language, raw_output="./verg/vergadering.txt", verbose=True)

    print("Aligning results")
    word_timestamps = align_results(file, vocal_target, whisper_results, info.language, device, verbose=True)

    print("Diarizing speakers")
    wsm, ssm = diarize_speakers(file, vocal_target, verbose=True)

    with open(f"{os.path.splitext(file)[0]}.txt", "w", encoding="utf-8-sig", errors="ignore") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(file)[0]}.srt", "w", encoding="utf-8-sig", errors="ignore") as srt:
        write_srt(ssm, srt)

