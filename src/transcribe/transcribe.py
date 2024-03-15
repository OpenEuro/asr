from sys import platform
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging
import torch
from transformers.utils import is_flash_attn_2_available
from pyannote.audio import Pipeline
from pyannote.core import Segment
import pandas as pd

languages = {
    "English": "en",
    "Chinese": "zh",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Polish": "pl",
    "Catalan": "ca",
    "Dutch": "nl",
    "Arabic": "ar",
    "Swedish": "sv",
    "Italian": "it",
    "Indonesian": "id",
    "Hindi": "hi",
    "Finnish": "fi",
    "Vietnamese": "vi",
    "Hebrew": "iw",
    "Ukrainian": "uk",
    "Greek": "el",
    "Malay": "ms",
    "Czech": "cs",
    "Romanian": "ro",
    "Danish": "da",
    "Hungarian": "hu",
    "Tamil": "ta",
    "Norwegian": "no",
    "Thai": "th",
    "Urdu": "ur",
    "Croatian": "hr",
    "Bulgarian": "bg",
    "Lithuanian": "lt",
    "Latin": "la",
    "Maori": "mi",
    "Malayalam": "ml",
    "Welsh": "cy",
    "Slovak": "sk",
    "Telugu": "te",
    "Persian": "fa",
    "Latvian": "lv",
    "Bengali": "bn",
    "Serbian": "sr",
    "Azerbaijani": "az",
    "Slovenian": "sl",
    "Kannada": "kn",
    "Estonian": "et",
    "Macedonian": "mk",
    "Breton": "br",
    "Basque": "eu",
    "Icelandic": "is",
    "Armenian": "hy",
    "Nepali": "ne",
    "Mongolian": "mn",
    "Bosnian": "bs",
    "Kazakh": "kk",
    "Albanian": "sq",
    "Swahili": "sw",
    "Galician": "gl",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Sinhala": "si",
    "Khmer": "km",
    "Shona": "sn",
    "Yoruba": "yo",
    "Somali": "so",
    "Afrikaans": "af",
    "Occitan": "oc",
    "Georgian": "ka",
    "Belarusian": "be",
    "Tajik": "tg",
    "Sindhi": "sd",
    "Gujarati": "gu",
    "Amharic": "am",
    "Yiddish": "yi",
    "Lao": "lo",
    "Uzbek": "uz",
    "Faroese": "fo",
    "Haitian creole": "ht",
    "Pashto": "ps",
    "Turkmen": "tk",
    "Nynorsk": "nn",
    "Maltese": "mt",
    "Sanskrit": "sa",
    "Luxembourgish": "lb",
    "Myanmar": "my",
    "Tibetan": "bo",
    "Tagalog": "tl",
    "Malagasy": "mg",
    "Assamese": "as",
    "Tatar": "tt",
    "Hawaiian": "haw",
    "Lingala": "ln",
    "Hausa": "ha",
    "Bashkir": "ba",
    "Javanese": "jw",
    "Sundanese": "su",
}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif platform == "darwin":
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res["chunks"]:
        start = item["timestamp"][0]
        end = item["timestamp"][1]
        text = item["text"]
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = "".join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = [".", "?", "!"]


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text

def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed

def make_conversation(transcribe_result, diarization_result):
    processed = diarize_text(transcribe_result, diarization_result)
    df = pd.DataFrame(processed, columns=["segment", "speaker", "text"])[
        ["speaker", "text"]
    ]
    df["key"] = (df["speaker"] != df["speaker"].shift(1)).astype(int).cumsum()
    conversation = df.groupby(["key", "speaker"])["text"].apply(" ".join).reset_index()
    conversation_list = list(zip(conversation.text, conversation.speaker))
    return conversation_list

# def transcriber(input: str, language: str, translate: bool, progress) -> dict:
def transcriber(input: str, model: str, language: str, translate: bool, diarize: bool, input_diarization_token) -> dict:
    """Transcribes the audio using the OpenAI Whisper model.
    Args:
        input: file path to the audio file in any format
        language: name of the language in which the audio is recorded
        translate: boolean indicator to enable immediate translation
    Returns: transcription and segment-timestamps.
    """
    model_id = model

    if diarize:

        pipeline_diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=input_diarization_token)

        # send pipeline to GPU (when available)
        pipeline_diarization.to(device)

        # apply pretrained pipeline
        diarization = pipeline_diarization(input)

    # print the result
    # for turn, _, speaker in diarization.itertracks(yield_label=True):
    #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        use_flash_attention_2=True if is_flash_attn_2_available() else False
    )

    print(device)

    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    language = languages.get(language, None)
    task = None
    if translate:
        task = "translate"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"task": task}
    )
    

    results = pipe(input)
    results["text"] = results["text"].strip()
 
    text = ""
    chunks = results.get("chunks", [])
    for chunk in chunks:
        text += chunk["text"] + "\n"

    # conversation = make_conversation(transcription, diarization)

    # Transform the list to skip one line each time
    # conversation_gradio = []
    # for i in range(0, len(conversation), 2):  # Increment by 2 to skip one line each time
    #     current_text = conversation[i][0]
    #     next_text = conversation[i + 1][0] if i + 1 < len(conversation) else ""
    #     conversation_gradio.append((current_text, next_text))

    return text
