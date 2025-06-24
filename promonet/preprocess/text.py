import string

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer

import promonet


###############################################################################
# Constants
###############################################################################


# Whisper model identifier
MODEL_ID = "openai/whisper-large-v3"


###############################################################################
# Whisper ASR
###############################################################################


import re
import unicodedata
import jaconv

class JapaneseTextNormalizer:
    """
    日本語のWER/CER評価用に、表記ゆれを整える正規化クラス
    """
    def __call__(self, text: str) -> str:
        # 全角→半角統一（英数字・記号など）
        text = unicodedata.normalize("NFKC", text)

        # 記号・句読点などを削除
        text = re.sub(r'[！!？?。、・（）「」『』【】［］〈〉《》―ー‐\-~"\'’‘“”=＋*･／/\\|‥#@※￥$%♯☆★◎〇×→←↑↓…‥]', '', text)

        # 数字を除去（または将来的に漢数字へ変換）
        text = re.sub(r'\d+', '', text)

        # カタカナ→ひらがな（音で比較しやすく）
        text = jaconv.kata2hira(text)

        # 空白・タブ・全角スペース削除
        text = re.sub(r'\s+', '', text)

        return text
    


def from_audio(audio, sample_rate=promonet.SAMPLE_RATE, gpu=None):
    """Perform ASR from audio"""
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    # Infer text
    results = infer(
        {
            'sampling_rate': sample_rate,
            'raw': audio.to(torch.float32).squeeze(dim=0).cpu().numpy()
        },
        gpu)

    # Lint
    return lint(results['text'])


def from_file(audio_file, gpu=None):
    """Perform Whisper ASR on an audio file"""
    # Infer text
    results = infer([str(audio_file)], gpu)

    # Lint
    return lint(results[0]['text'])


def from_file_to_file(audio_file, output_file, gpu=None):
    """Perform Whisper ASR and save"""
    from_files_to_files([audio_file], [output_file], gpu)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Perform batched Whisper ASR from files and save"""
    # Infer text
    results = infer([str(audio_file) for audio_file in audio_files], gpu)

    # Lint
    results = [lint(result['text']) for result in results]

    # Save
    for result, output_file in zip(results, output_files):
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(result)


###############################################################################
# Utilities
###############################################################################


def infer(audio, gpu=None):
    """Batched Whisper ASR"""
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    # Cache model
    if not hasattr(infer, 'pipe') or infer.device != device:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        #japanese ASR requires forced decoder ids
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="japanese", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        
        infer.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=64,
            return_timestamps=False,
            torch_dtype=torch.float16,
            device=device,
            generate_kwargs={
            "language": "ja",
            })
        infer.device = device

    return infer.pipe(audio)


def lint(text):
    """日本語WER/CER評価用にテキストを整形する"""
    if not hasattr(lint, 'normalizer'):
        lint.normalizer = JapaneseTextNormalizer()
    return lint.normalizer(text)
