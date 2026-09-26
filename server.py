import sys
import io
import re
import json
import os
import difflib
import urllib.request
import soundfile as sf
import numpy as np
from difflib import SequenceMatcher
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import uvicorn
import eng_to_ipa as ipa_engine

app = FastAPI(title="Ultra-Light Pure ONNX Phonetics Engine")

# CORS Bypass
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚡ Cron-Job और Health Check के लिए हल्का एंडपॉइंट (Cold Start रोकने के लिए)
@app.get("/")
def home():
    return {"status": "Pronounce AI Server Running 24/7", "engine": "Pure ONNX INT8"}

# ⚡ Wav2Vec2 Vocabulary Mapping
VOCAB = [
    "<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", 
    "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"
]

def ctc_decode(predictions):
    pred_ids = np.argmax(predictions, axis=-1)[0]
    decoded_chars = []
    prev_id = -1
    for p in pred_ids:
        if p != prev_id and p != 0: # 0 is <pad>
            if p < len(VOCAB):
                char = VOCAB[p]
                decoded_chars.append(" " if char == "|" else char)
        prev_id = p
    return "".join(decoded_chars).strip()

# ⚡ Pre-quantized Ultra-Light ONNX (~95MB)
ONNX_FILE = os.path.join(os.path.dirname(__file__), "wav2vec2_model.onnx")
ONNX_URL = "https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/main/onnx/model_quantized.onnx"

if not os.path.exists(ONNX_FILE):
    print("Pre-quantized ONNX model download ho raha hai...", flush=True)
    urllib.request.urlretrieve(ONNX_URL, ONNX_FILE)
    print("Download complete!", flush=True)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
ort_session = ort.InferenceSession(ONNX_FILE, sess_options, providers=['CPUExecutionProvider'])
print("[SUCCESS] Pure ONNX Engine Safaltapurvak Load Hua (RAM under 100MB)!", flush=True)

JSON_PATH = os.path.join(os.path.dirname(__file__), "phonetics_rules.json")
CUSTOM_RULES = {}
RAW_CATEGORIES = []

def load_rules_from_json():
    global CUSTOM_RULES, RAW_CATEGORIES
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                content = json.load(f)
                RAW_CATEGORIES = content.get("categories", [])
                for cat in RAW_CATEGORIES:
                    cat_name = cat.get("name", "")
                    words_dict = cat.get("words", {})
                    for word_key, details in words_dict.items():
                        w = word_key.strip().upper().split("_")[0]
                        raw_wrongs = [x.strip().upper() for x in details.get("wrong", [])]
                        CUSTOM_RULES[w] = {
                            "category": cat_name,
                            "ipa": details.get("ipa", ""),
                            "note": details.get("note", ""),
                            "wrong": raw_wrongs,
                            "silent_letter": details.get("silent_letter", False)
                        }
            print(f"[SUCCESS] Kul {len(CUSTOM_RULES)} rules load huye!", flush=True)
        except Exception as err:
            print(f"[ERROR] JSON error: {err}", flush=True)
    else:
        print("[WARNING] 'phonetics_rules.json' nahi mili.", flush=True)

load_rules_from_json()

def resample_to_16k(audio_data: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == 16000 or len(audio_data) == 0:
        return audio_data.astype(np.float32)
    target_len = int(round(len(audio_data) * 16000 / orig_sr))
    resampled = np.interp(
        np.linspace(0, len(audio_data), target_len, endpoint=False),
        np.arange(len(audio_data)),
        audio_data
    )
    return resampled.astype(np.float32)

def get_acoustic_spectral_ratio(audio_chunk, sample_rate=16000):
    if len(audio_chunk) < 200:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(audio_chunk))
    freqs = np.fft.rfftfreq(len(audio_chunk), 1.0 / sample_rate)
    low_band = np.sum(fft_vals[(freqs >= 300) & (freqs <= 1200)] ** 2) + 1e-8
    high_band = np.sum(fft_vals[(freqs >= 1800) & (freqs <= 3200)] ** 2) + 1e-8
    return float(high_band / low_band)

def get_sibilant_spectral_analysis(audio_chunk, sample_rate=16000):
    if len(audio_chunk) < 200:
        return 0.0, 0.0
    fft_vals = np.abs(np.fft.rfft(audio_chunk))
    freqs = np.fft.rfftfreq(len(audio_chunk), 1.0 / sample_rate)
    sh_band = np.sum(fft_vals[(freqs >= 2500) & (freqs <= 4500)] ** 2) + 1e-8
    s_band = np.sum(fft_vals[(freqs >= 5200) & (freqs <= 7800)] ** 2) + 1e-8
    s_vs_sh_ratio = float(s_band / sh_band)
    zero_crossings = np.nonzero(np.diff(audio_chunk > 0))[0]
    zcr = float(len(zero_crossings) / len(audio_chunk))
    return s_vs_sh_ratio, zcr

@app.get("/get-drills")
async def get_drills():
    drill_list = []
    phonetics_dict = {}
    for cat in RAW_CATEGORIES:
        drill_list.append({
            "id": cat.get("id"),
            "name": cat.get("name"),
            "sentence": cat.get("drill_sentence", "")
        })
        for w_key, w_val in cat.get("words", {}).items():
            clean_key = w_key.split("_")[0].lower()
            phonetics_dict[clean_key] = {
                "ipa": w_val.get("ipa", ""),
                "note": w_val.get("note", ""),
                "pitch": [40, 45],
                "jaw": 0.55 if any(x in clean_key.upper() for x in ["MAN", "BAT", "PAN", "CALM", "AND", "ON"]) else 0.25,
                "tongueX": 0.2,
                "tongueY": 0.7 if any(x in clean_key.upper() for x in ["VISION", "PLEASURE", "ACTION", "SPECIAL"]) else 0.35,
                "lips": 0.75 if (clean_key.upper().startswith("W") or any(x in clean_key.upper() for x in ["WATER", "WINDOW", "WERE", "WAS"])) else 0.1
            }
    return {"drills": drill_list, "phonetics": phonetics_dict}

@app.get("/get-word-ipa")
async def get_word_ipa(word: str):
    clean_w = word.strip().lower()
    if clean_w.upper() in CUSTOM_RULES and CUSTOM_RULES[clean_w.upper()].get("ipa"):
        return {"word": word, "ipa": CUSTOM_RULES[clean_w.upper()]["ipa"]}
    generated_ipa = ipa_engine.convert(clean_w)
    return {"word": word, "ipa": f"/{generated_ipa}/"}

def evaluate_word_phonetics(ref_clean: str, spoken_clean: str, next_word: str = "", spectral_ratio: float = 0.0, audio_chunk: np.ndarray = None):
    if not spoken_clean or spoken_clean == "[छूट गया]":
        return False, "यह शब्द पढ़ने में छूट गया।"

    ref_clean = ref_clean.upper().strip()
    spoken_clean = spoken_clean.upper().strip()

    s_sh_ratio, zcr = (0.0, 0.0)
    if audio_chunk is not None and len(audio_chunk) > 200:
        s_sh_ratio, zcr = get_sibilant_spectral_analysis(audio_chunk)

    if ref_clean in CUSTOM_RULES:
        rule = CUSTOM_RULES[ref_clean]
        if spoken_clean in rule["wrong"]:
            return False, f"त्रुटि [{rule['category']}]: {rule['note']}"

    if any(x in ref_clean for x in ["SH", "TION", "SION", "TIOUS", "TIENT", "CIAL"]):
        if ("S" in spoken_clean and "SH" not in spoken_clean) or s_sh_ratio > 1.35 or spoken_clean.endswith("SAN"):
            return False, "ध्वनि त्रुटि: आपने 'श' (/ʃ/) की जगह 'स' (/s/) बोल दिया है।"

    if ("Z" in ref_clean or "SE" in ref_clean) and ("J" in spoken_clean or spoken_clean.startswith("G")):
        return False, "ध्वनि त्रुटि: 'ज' नहीं, गले में कम्पन के साथ 'ज़' (/z/) बोलें।"

    if ref_clean == "PRONUNCIATION":
        if spoken_clean in ["PRDN", "PRONUNCIASAN", "PRONOUNCIATION"] or spoken_clean.endswith("SAN"):
            return False, "ध्वनि त्रुटि: 'प्र-नन-सी-एशन' बोलें, 'प्रदन' या 'सन' नहीं।"

    similarity = SequenceMatcher(None, ref_clean, spoken_clean).ratio()
    if ref_clean == spoken_clean or similarity >= 0.85:
        return True, ""

    return False, f"सुना गया: '{spoken_clean}', सही शब्द: '{ref_clean}'"

@app.post("/verify-pronunciation")
async def verify_pronunciation(
    audio: UploadFile = File(...),
    reference_text: str = Form(...)
):
    ref_tokens = [w for w in re.sub(r'[^A-Za-z\s]', '', reference_text).upper().split() if w]

    try:
        audio_bytes = await audio.read()
        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        return {
            "transcription": "",
            "results": [
                {"index": idx, "word": ref_word, "is_correct": False, "heard": "[खाली]", "error_detail": "ऑडियो लोड नहीं हुआ।"}
                for idx, ref_word in enumerate(ref_tokens)
            ]
        }

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = resample_to_16k(data, sample_rate)

    if len(data) < 400:
        return {
            "transcription": "",
            "results": [
                {"index": idx, "word": ref_word, "is_correct": False, "heard": "[खाली]", "error_detail": "ऑडियो बहुत छोटा था।"}
                for idx, ref_word in enumerate(ref_tokens)
            ]
        }

    spectral_ratio = get_acoustic_spectral_ratio(data)

    pad_samples = int(16000 * 0.25)
    data_padded = np.pad(data, (pad_samples, pad_samples), mode='constant', constant_values=0)
    max_val = np.max(np.abs(data_padded))
    if max_val > 0.01:
        data_padded = (data_padded / max_val) * 0.95

    # ⚡ Pure ONNX Fast Inference
    input_values = np.expand_dims(data_padded.astype(np.float32), axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: input_values}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]

    transcription = ctc_decode(logits).upper()
    spoken_tokens = [w for w in re.sub(r'[^A-Z\s]', '', transcription).split() if w]

    print(f"\n[ACOUSTIC LOG] Target: '{reference_text}' | Heard: '{transcription}'", flush=True)

    matcher = difflib.SequenceMatcher(None, ref_tokens, spoken_tokens)
    aligned_spoken = [None] * len(ref_tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('equal', 'replace'):
            for r_idx, s_idx in zip(range(i1, i2), range(j1, j2)):
                aligned_spoken[r_idx] = spoken_tokens[s_idx]

    eval_results = []
    for idx, ref_word in enumerate(ref_tokens):
        next_w = ref_tokens[idx + 1] if idx + 1 < len(ref_tokens) else ""
        spoken_w = aligned_spoken[idx]

        if spoken_w is None:
            eval_results.append({
                "index": idx,
                "word": ref_word,
                "is_correct": False,
                "heard": "[छूट गया]",
                "error_detail": "यह शब्द पढ़ने में छूट गया।"
            })
            continue

        is_valid, err_msg = evaluate_word_phonetics(ref_word, spoken_w, next_w, spectral_ratio, data)
        eval_results.append({
            "index": idx,
            "word": ref_word,
            "is_correct": is_valid,
            "heard": spoken_w,
            "error_detail": err_msg
        })

    return {
        "transcription": transcription,
        "results": eval_results
    }

@app.post("/verify-carrier-framed-word")
async def verify_carrier_framed_word(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    carrier_sentence: str = Form(...)
):
    target = target_word.strip().upper()
    carrier_tokens = [w for w in re.sub(r'[^A-Za-z\s]', '', carrier_sentence).upper().split() if w]

    try:
        audio_bytes = await audio.read()
        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        data = resample_to_16k(data, sample_rate)
    except Exception as e:
        return {"word": target, "is_correct": False, "score": 0, "error_detail": "ऑडियो लोड नहीं हुआ।"}

    max_val = np.max(np.abs(data))
    if max_val > 0.01:
        data = (data / max_val) * 0.95
    else:
        return {"word": target, "is_correct": False, "score": 0, "error_detail": "आवाज़ बहुत धीमी या शांत थी।"}

    pad_samples = int(16000 * 0.20)
    data_padded = np.pad(data, (pad_samples, pad_samples), mode='constant', constant_values=0)

    input_values = np.expand_dims(data_padded.astype(np.float32), axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: input_values}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]

    transcription = ctc_decode(logits).upper()
    spoken_tokens = [w for w in re.sub(r'[^A-Z\s]', '', transcription).split() if w]
    print(f"\n[CARRIER CHECK] Sentence: '{carrier_sentence}' | Target: '{target}' | Heard: '{transcription}'", flush=True)

    if not spoken_tokens:
        return {"word": target, "is_correct": False, "score": 0, "error_detail": "कोई शब्द सुनाई नहीं दिया।"}

    target_idx = -1
    for idx, w in enumerate(carrier_tokens):
        if w == target:
            target_idx = idx
            break

    matcher = difflib.SequenceMatcher(None, carrier_tokens, spoken_tokens)
    target_spoken = None

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i1 <= target_idx < i2:
            offset = target_idx - i1
            if j1 + offset < j2:
                target_spoken = spoken_tokens[j1 + offset]
            elif j1 < j2:
                target_spoken = spoken_tokens[j1]
            break

    if not target_spoken:
        best_sim = 0
        for spk in spoken_tokens:
            sim = SequenceMatcher(None, target, spk).ratio()
            if sim > best_sim:
                best_sim = sim
                target_spoken = spk

    target_spoken = target_spoken or ""
    similarity = SequenceMatcher(None, target, target_spoken).ratio()
    score = int(similarity * 100)

    is_valid, err_msg = evaluate_word_phonetics(target, target_spoken, "", 0.0, data)

    if similarity < 0.80:
        is_valid = False
        err_msg = f"सुना गया: '{target_spoken if target_spoken else '[अस्पष्ट]'}' | सही शब्द: '{target}'"

    return {
        "word": target,
        "is_correct": is_valid,
        "score": score if is_valid else min(score, 45),
        "heard_target": target_spoken,
        "error_detail": "" if is_valid else err_msg
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
