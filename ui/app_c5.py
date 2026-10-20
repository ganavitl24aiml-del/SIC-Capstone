from flask import (
    Flask,
    jsonify,
    send_from_directory,
    render_template_string,
    abort,
    request
)

from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
from functools import lru_cache
import pandas as pd
import os
import sys
import re
import pickle
import requests
from collections import Counter

try:
    import numpy as np
    from PIL import Image, ImageStat
except Exception:
    np = None
    Image = None
    ImageStat = None

app = Flask(__name__)

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CLIENTS_DIR = ROOT / "data" / "chest_xray" / "client_visualization"

DISEASES = [
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Nodule",
    "Infiltration"
]

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLIENTS = {}


# ============================================================
# LOAD HOSPITAL DATA
# ============================================================

def load_clients():

    CLIENTS.clear()

    for cid in range(1, 6):

        folder = CLIENTS_DIR / f"client_{cid}"
        csv = folder / "labels.csv"

        if not csv.exists():
            continue

        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"Could not read {csv}: {e}")
            continue

        imgdir = folder / "images"

        if not imgdir.exists():
            imgdir = folder

        if not imgdir.exists():
            files = set()
        else:
            files = {
                p.name
                for p in imgdir.iterdir()
                if p.is_file()
                and p.suffix.lower() in EXTS
            }

        CLIENTS[cid] = {
            "df": df,
            "dir": imgdir,
            "files": files
        }


load_clients()


# ============================================================
# DATA HELPERS
# ============================================================

def counts(df):

    return {
        disease: int(df[disease].sum())
        if disease in df.columns else 0
        for disease in DISEASES
    }


def labels(row):

    output = []

    for disease in DISEASES:

        try:
            if int(row[disease]) == 1:
                output.append(disease)

        except (KeyError, ValueError, TypeError):
            pass

    return output


def get_client_images(cid):

    if cid not in CLIENTS:
        return []

    return CLIENTS[cid]["files"]


# ============================================================
# FEDAVG MODEL
# ============================================================

prediction_error = None

try:

    INTEGRATION_DIR = ROOT / "integration"

    if str(INTEGRATION_DIR) not in sys.path:
        sys.path.insert(0, str(INTEGRATION_DIR))

    from prediction import (
        load_fedavg_model,
        load_tuned_thresholds,
        predict_xray,
        MODEL_PATH
    )

    print("✓ FedAvg prediction module loaded")

except Exception as e:

    prediction_error = str(e)

    print("⚠ FedAvg prediction module could not be loaded:")
    print(e)


@lru_cache(maxsize=1)
def get_model():

    if prediction_error:
        raise RuntimeError(prediction_error)

    return load_fedavg_model()


@lru_cache(maxsize=1)
def get_thresholds():

    if prediction_error:
        raise RuntimeError(prediction_error)

    return load_tuned_thresholds()


# ============================================================
# RAG
# ============================================================
RAG_INDEX_PATH = ROOT / "rag" / "vector_db" / "medical_index.faiss"
RAG_CHUNKS_PATH = ROOT / "rag" / "vector_db" / "chunks.pkl"


rag_error = None

try:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from integration.rag_pipeline import ask_medical_rag

    print("✓ RAG retrieval pipeline loaded")

except Exception as e:
    rag_error = str(e)
    ask_medical_rag = None
    print("⚠ RAG embedding pipeline unavailable; local retrieval will be used:")
    print(e)


# Ollama is used ONLY for answer generation. Retrieval is always grounded in
# the project's saved medical documents.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "90"))


DISEASE_ALIASES = {
    "pneumonia": ["pneumonia", "pneumonias", "lung infection", "infection"],
    "atelectasis": ["atelectasis", "collapse of lung", "lung collapse"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart enlargement"],
    "effusion": ["effusion", "pleural effusion", "fluid around the lung", "fluid in pleural space"],
    "nodule": ["nodule", "pulmonary nodule", "lung nodule"],
    "infiltration": ["infiltration", "infiltrate", "pulmonary infiltrate"]
}


def _normalise_disease(value):
    value = str(value or "").strip().lower()
    for disease, aliases in DISEASE_ALIASES.items():
        if value == disease or any(alias in value for alias in aliases):
            return disease
    return ""


def _question_disease(question):
    q = str(question).lower()
    for disease, aliases in DISEASE_ALIASES.items():
        if any(re.search(r"\b" + re.escape(alias) + r"\b", q) for alias in aliases):
            return disease
    return ""


def _chunk_source(chunk):
    return str(
        chunk.get("filename")
        or chunk.get("source")
        or chunk.get("file")
        or chunk.get("name")
        or "Unknown source"
    )


def _chunk_disease(chunk):
    explicit = _normalise_disease(
        chunk.get("disease") or chunk.get("label") or chunk.get("category")
    )
    if explicit:
        return explicit

    source = _chunk_source(chunk).lower()
    for disease, aliases in DISEASE_ALIASES.items():
        if any(alias.replace(" ", "_") in source or alias in source for alias in aliases):
            return disease

    return ""


def _expand_query(question):
    """Add medically relevant synonyms without adding outside medical facts."""
    q = str(question).strip().lower()
    disease = _question_disease(q)
    terms = [q]

    if disease:
        terms.append(disease)
        terms.extend(DISEASE_ALIASES.get(disease, []))

    if "symptom" in q or "sign" in q or "clinical" in q:
        terms.extend(["symptoms", "clinical manifestations", "clinical features"])

    if "x-ray" in q or "xray" in q or "radiograph" in q:
        terms.extend(["chest radiograph", "chest x-ray", "radiological findings"])

    return " ".join(dict.fromkeys(terms))


def _load_saved_chunks():
    if not RAG_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"RAG chunks file not found: {RAG_CHUNKS_PATH}")

    with open(RAG_CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    if isinstance(chunks, dict):
        chunks = chunks.get("chunks", chunks.get("documents", []))

    if not isinstance(chunks, (list, tuple)):
        raise ValueError("Saved RAG chunks must be a list of document chunks.")

    return chunks


def _lexical_rag(question, k=6):
    """Retrieve relevant chunks from the project's saved medical corpus.

    This deliberately does not invent medical content. It ranks the project's
    own chunks, with a strong boost for the disease explicitly asked about.
    """
    chunks = _load_saved_chunks()
    expanded = _expand_query(question)
    q_tokens = set(re.findall(r"[a-zA-Z]{3,}", expanded))
    disease = _question_disease(question)

    stop = {
        "what", "are", "is", "the", "a", "an", "of", "to", "in", "on",
        "for", "and", "or", "with", "from", "can", "how", "does", "do",
        "about", "common", "clinical", "features", "feature", "please",
        "explain", "tell", "me", "which", "this", "that", "be", "seen",
        "what", "their", "they", "does", "why", "when", "where", "who",
        "disease", "information", "related"
    }
    q_tokens = {t for t in q_tokens if t not in stop}

    scored = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        text = str(chunk.get("text", chunk.get("content", ""))).strip()
        if not text:
            continue

        source = _chunk_source(chunk)
        chunk_disease = _chunk_disease(chunk)
        hay = (source + " " + text).lower()
        tokens = set(re.findall(r"[a-zA-Z]{3,}", hay))

        overlap = len(q_tokens & tokens)
        phrase_hits = sum(1 for token in q_tokens if token in hay)
        score = overlap * 4 + phrase_hits

        # Disease-specific questions must strongly prefer documents belonging
        # to that disease, while still allowing relevant cross-disease sources.
        if disease:
            if chunk_disease == disease:
                score += 30
            elif disease in hay:
                score += 12
            else:
                score -= 4

        # Symptom questions should prefer passages containing symptom language.
        if any(word in question.lower() for word in ("symptom", "sign", "clinical")):
            score += sum(
                2 for word in (
                    "symptom", "symptoms", "clinical", "manifestation",
                    "presented", "cough", "fever", "pain", "dyspnea",
                    "shortness", "breath"
                ) if word in hay
            )

        if score > 0:
            scored.append((score, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))

    selected = []
    used_sources = set()
    for score, _, chunk in scored:
        source = _chunk_source(chunk)
        source_key = source.lower()
        if source_key in used_sources:
            continue
        used_sources.add(source_key)
        selected.append({
            "source": source,
            "text": str(chunk.get("text", chunk.get("content", ""))).strip(),
            "disease": _chunk_disease(chunk),
            "score": score
        })
        if len(selected) >= k:
            break

    return selected


def _clean_sources_for_model(sources, max_chars_per_source=5000, max_total_chars=18000):
    """Build a bounded context window for Ollama."""
    blocks = []
    total = 0

    for i, src in enumerate(sources, start=1):
        text = " ".join(str(src.get("text", "")).split())
        if not text:
            continue
        text = text[:max_chars_per_source]
        block = (
            f"SOURCE {i}: {src.get('source', 'Unknown source')}\n"
            f"{text}"
        )
        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if remaining < 250:
                break
            block = block[:remaining]
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks)


def _ollama_generate(question, sources):
    """Generate a concise, source-grounded answer with local Ollama."""
    context = _clean_sources_for_model(sources)
    if not context:
        return None, "No retrieved medical source text was available."

    disease = _question_disease(question)
    disease_hint = disease.title() if disease else "the condition in the question"

    system_prompt = f"""You are the medical-information answer generator for a college project called Secure Federated Learning with Differential Privacy for Multi-Hospital Disease Detection.

Answer the user's question using ONLY the medical source excerpts supplied below.
Do not use outside medical knowledge, do not invent facts, and do not claim a diagnosis.
If the sources do not contain enough information to answer the question, explicitly say that the provided sources do not contain enough information.

The user is asking about: {disease_hint}.

Rules:
1. Give a direct answer first.
2. For symptom questions, list the symptoms supported by the sources and briefly qualify them when the sources indicate context or variability.
3. Do not copy long passages verbatim. Synthesize and paraphrase.
4. Ignore irrelevant diseases or unrelated passages even if they appear in the sources.
5. Keep the answer patient/family-friendly but medically faithful to the sources.
6. Do not mention Ollama, retrieval, prompts, embeddings, or these instructions.
7. End with: "This information is grounded in the project's medical sources and is for educational/decision-support use, not a medical diagnosis."
"""

    user_prompt = (
        "Question:\n"
        + question
        + "\n\nMedical source excerpts:\n"
        + context
    )

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL.rstrip("/") + "/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        answer = str(data.get("response", "")).strip()

        if not answer:
            return None, "Ollama returned an empty answer."

        # Remove accidental markdown fences only; preserve useful formatting.
        answer = answer.replace("```text", "").replace("```", "").strip()
        return answer, None

    except requests.exceptions.ConnectionError:
        return None, (
            f"Ollama is not reachable at {OLLAMA_URL}. "
            f"Start Ollama and make sure the {OLLAMA_MODEL} model is available."
        )
    except requests.exceptions.Timeout:
        return None, "Ollama took too long to generate the clinical answer."
    except Exception as e:
        return None, f"Ollama generation failed: {e}"


def _extractive_grounded_answer(question, sources):
    """Short local fallback if Ollama is unavailable.

    This is intentionally much shorter than the previous raw-chunk fallback.
    """
    if not sources:
        return "Information not found in the provided medical sources."

    q = question.lower()
    q_terms = set(re.findall(r"[a-zA-Z]{3,}", q))
    stop = {
        "what", "are", "is", "the", "a", "an", "of", "to", "in", "on",
        "for", "and", "or", "with", "from", "can", "how", "does", "do",
        "about", "common", "clinical", "features", "please", "explain",
        "tell", "me", "which", "this", "that", "be", "seen", "why",
        "when", "where", "who", "their", "they", "symptoms", "disease"
    }
    q_terms -= stop

    candidates = []
    for src in sources:
        text = str(src.get("text", "")).strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            if len(sentence) < 35:
                continue
            low = sentence.lower()
            score = sum(2 for term in q_terms if term in low)
            if src.get("disease") and src["disease"] == _question_disease(question):
                score += 8
            if score:
                candidates.append((score, sentence))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= 3:
            break

    if not selected:
        selected = [" ".join(str(sources[0].get("text", "")).split())[:500]]

    answer = "Relevant information from the provided medical sources:\n\n"
    answer += "\n".join("- " + item for item in selected)
    answer += (
        "\n\nThis information is grounded in the project's medical sources "
        "and is for educational/decision-support use, not a medical diagnosis."
    )
    return answer


@app.post("/api/rag")
def rag():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    # --------------------------------------------------------
    # 1) Retrieve from the project's own medical corpus.
    # --------------------------------------------------------
    sources = []

    # Prefer the saved embedding/FAISS pipeline when it is available and
    # returns actual source passages. We do NOT accept an Ollama error as the
    # final answer from that pipeline; Ollama generation is handled here.
    try:
        if ask_medical_rag is not None and rag_error is None:
            result = ask_medical_rag(question, k=6)
            candidate_sources = result.get("sources", []) if isinstance(result, dict) else []
            if candidate_sources:
                for src in candidate_sources:
                    if isinstance(src, str):
                        sources.append({"source": src, "text": src, "disease": ""})
                    elif isinstance(src, dict):
                        sources.append({
                            "source": str(src.get("source") or src.get("filename") or "Unknown source"),
                            "text": str(src.get("text") or src.get("content") or ""),
                            "disease": _normalise_disease(src.get("disease"))
                        })
    except Exception as e:
        print("Embedding RAG retrieval unavailable; using saved chunks:", e)

    # The local lexical retriever is also used to correct weak disease matches
    # and to guarantee operation when Hugging Face/FAISS cannot initialize.
    try:
        local_sources = _lexical_rag(question, k=6)
        if local_sources:
            # Disease-focused local retrieval is authoritative for the final
            # context when the question names one of the six project diseases.
            disease = _question_disease(question)
            if disease:
                disease_sources = [s for s in local_sources if s.get("disease") == disease]
                if disease_sources:
                    sources = disease_sources + [
                        s for s in local_sources if s not in disease_sources
                    ]
            elif not sources:
                sources = local_sources
    except Exception as e:
        print("Saved-chunk RAG retrieval unavailable:", e)

    if not sources:
        return jsonify({
            "question": question,
            "answer": "Information not found in the provided medical sources.",
            "sources": [],
            "grounded": False,
            "method": "Medical source retrieval"
        })

    # Remove duplicate sources and empty passages before generation.
    unique = []
    seen = set()
    for src in sources:
        source = str(src.get("source", "Unknown source"))
        text = str(src.get("text", "")).strip()
        key = (source.lower(), text[:200].lower())
        if not text or key in seen:
            continue
        seen.add(key)
        unique.append({
            "source": source,
            "text": text,
            "disease": src.get("disease", "")
        })

    sources = unique[:6]

    # --------------------------------------------------------
    # 2) Generate a concise answer with local Ollama.
    # --------------------------------------------------------
    answer, generation_error = _ollama_generate(question, sources)

    if answer:
        return jsonify({
            "question": question,
            "answer": answer,
            "sources": [
                {"source": s["source"], "text": s["text"]}
                for s in sources
            ],
            "grounded": True,
            "method": f"Medical source retrieval + Ollama {OLLAMA_MODEL}"
        })

    # --------------------------------------------------------
    # 3) Safe fallback: never expose the Ollama error as the answer.
    # --------------------------------------------------------
    fallback = _extractive_grounded_answer(question, sources)
    return jsonify({
        "question": question,
        "answer": fallback,
        "sources": [
            {"source": s["source"], "text": s["text"]}
            for s in sources
        ],
        "grounded": True,
        "method": "Saved medical chunks + local fallback",
        "ollama_status": generation_error
    })


# ============================================================
# API: OVERVIEW
# ============================================================

@app.get("/api/overview")
def overview():

    return jsonify({

        "project": "Secure Federated Learning with Differential Privacy for Multi-Hospital Disease Detection",

        "hospitals": len(CLIENTS),

        "diseases": len(DISEASES),

        "assigned": sum(
            len(client["df"])
            for client in CLIENTS.values()
        ),

        "aggregation": "FedAvg",

        "model": "ResNet18",

        "privacy": "Differential Privacy proof-of-concept",

        "global_model": (
            "Loaded"
            if prediction_error is None
            else "Unavailable"
        ),

        "rag": (
            "Online"
            if rag_error is None
            else "Unavailable"
        ),

        "clients": [

            {
                "id": cid,

                "name": f"Hospital {cid:02d}",

                "assigned": len(client["df"]),

                "available": len(client["files"]),

                "images": len(client["files"]),

                "missing": max(
                    0,
                    len(client["df"]) -
                    len(client["files"])
                ),

                "diseases": counts(client["df"])

            }

            for cid, client
            in sorted(CLIENTS.items())
        ]
    })


# ============================================================
# API: CLIENT DATA
# ============================================================

@app.get("/api/client/<int:cid>")
def client(cid):

    if cid not in CLIENTS:
        abort(404)

    x = CLIENTS[cid]

    disease = request.args.get(
        "disease",
        "Pneumonia"
    )

    if disease not in DISEASES:
        disease = "Pneumonia"

    try:

        limit = min(
            max(
                int(request.args.get("limit", "16")),
                1
            ),
            64
        )

    except ValueError:

        limit = 16

    df = x["df"]

    matching = df[
        df[disease] == 1
    ]

    images = []

    for _, row in matching.iterrows():

        name = str(
            row["Image Index"]
        )

        if name not in x["files"]:
            continue

        images.append({

            "name": name,

            "labels": labels(row),

            "url":
                f"/image/{cid}/{name}"

        })

        if len(images) >= limit:
            break

    available_matching = sum(
        str(name) in x["files"]
        for name in matching["Image Index"]
    )

    return jsonify({

        "id": cid,

        "name": f"Hospital {cid:02d}",

        "assigned": len(df),

        "available": len(x["files"]),

        "missing": max(
            0,
            len(df) - len(x["files"])
        ),

        "diseases": counts(df),

        "disease": disease,

        "available_matching":
            available_matching,

        "images": images
    })


# ============================================================
# API: ACTUAL FEDAVG PREDICTION
# ============================================================

@app.get("/api/predict/<int:cid>/<path:name>")
# ============================================================
# API: UPLOAD NEW X-RAY AND RUN FEDAVG MODEL
# ============================================================

def validate_uploaded_xray(image_path):
    """Reject obvious non-X-ray/blank images before the trained model runs.

    The saved FedAvg classifier is a disease classifier, not an image-type
    detector. Without this guard, a white page or arbitrary photograph can
    still receive a disease probability because the neural network always
    produces six sigmoid outputs.
    """
    if Image is None or np is None:
        return True, ""

    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32)

        if arr.size == 0:
            return False, "The uploaded image is empty. Please upload a chest X-ray."

        mean = float(arr.mean())
        std = float(arr.std())
        near_white = float((arr >= 250).mean())
        near_black = float((arr <= 5).mean())

        # Obvious blank-page / blank-image rejection.
        if (mean >= 247 and std <= 7) or near_white >= 0.97 or near_black >= 0.97:
            return False, (
                "This image appears to be blank or nearly blank, so the "
                "FedAvg disease model was not run. Please upload a chest X-ray."
            )

        # A genuine radiograph normally contains substantially more tonal
        # variation than a plain document/page. Keep this conservative so
        # legitimate low-contrast X-rays are not rejected unnecessarily.
        if std < 4.0:
            return False, (
                "The uploaded image has insufficient radiographic detail. "
                "Please upload a clear chest X-ray."
            )

        return True, ""

    except Exception as e:
        return False, f"Unable to read the uploaded image: {e}"


@app.post("/api/upload-predict")
def upload_predict():

    if prediction_error:
        return jsonify({
            "error": "FedAvg prediction module is unavailable.",
            "details": prediction_error
        }), 500

    if "image" not in request.files:
        return jsonify({
            "error": "No X-ray image was uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "Please select an X-ray image."
        }), 400

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed:
        return jsonify({
            "error":
                "Please upload JPG, JPEG, PNG, WEBP or BMP."
        }), 400

    safe_name = secure_filename(
        file.filename
    )

    temp_path = None

    try:

        # Save temporarily.
        # The uploaded image is NOT added to any hospital dataset.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:

            temp_path = temp.name

        file.save(temp_path)

        valid_image, validation_message = validate_uploaded_xray(temp_path)
        if not valid_image:
            return jsonify({
                "success": False,
                "error": validation_message
            }), 422

        # IMPORTANT: this route loads ONLY the already-saved global FedAvg
        # checkpoint; it does not retrain, fine-tune, or substitute another model.
        model, device = get_model()

        # Load experimentally tuned thresholds
        thresholds = get_thresholds()

        # Run prediction using your existing prediction.py
        results = predict_xray(
            image_path=temp_path,
            model=model,
            device=device,
            thresholds=thresholds
        )

        return jsonify({

            "success": True,

            "filename": safe_name,

            "model": "FedAvg ResNet18",

            "model_path": str(MODEL_PATH),

            "diseases": results

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                f"Prediction failed: {str(e)}"

        }), 500

    finally:

        # Remove temporary uploaded image
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
def predict(cid, name):

    if cid not in CLIENTS:
        abort(404)

    name = Path(name).name

    if name not in CLIENTS[cid]["files"]:
        abort(404)

    image_path = CLIENTS[cid]["dir"] / name

    try:

        model, device = get_model()

        thresholds = get_thresholds()

        results = predict_xray(
            image_path=str(image_path),
            model=model,
            device=device,
            thresholds=thresholds
        )

        return jsonify({

            "hospital":
                f"Hospital {cid:02d}",

            "image": name,

            "model": "FedAvg ResNet18",

            "results": results

        })

    except Exception as e:

        return jsonify({

            "error":
                f"Prediction failed: {str(e)}"

        }), 500


# ============================================================
# IMAGE SERVING
# ============================================================

@app.get("/image/<int:cid>/<path:name>")
def image(cid, name):

    if cid not in CLIENTS:
        abort(404)

    name = Path(name).name

    if name not in CLIENTS[cid]["files"]:
        abort(404)

    return send_from_directory(
        CLIENTS[cid]["dir"],
        name
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

DASHBOARD_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Medical Assistant • FL-RAG</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
:root{--navy:#0A2540;--navy2:#071b2e;--blue:#2563eb;--cyan:#0ea5e9;--green:#16a34a;--red:#dc2626;--ink:#172033;--muted:#64748b;--line:#dbe3ec;--work:#f3f6f9}
*{box-sizing:border-box} body{margin:0;background:#061525;color:#e8eef5;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{min-height:100vh;background:#061525}.utility{height:36px;background:#061a2c;border-bottom:1px solid rgba(255,255,255,.08);display:flex;align-items:center;justify-content:space-between;padding:0 22px;font-size:11px;color:#9fb2c7}.utility-actions{display:flex;gap:8px}.utility-actions span{padding:4px 9px;border:1px solid rgba(255,255,255,.08);border-radius:5px}.header{background:var(--navy);border-bottom:1px solid rgba(255,255,255,.1);padding:14px 22px;display:grid;grid-template-columns:290px 1fr auto;gap:22px;align-items:center}.brand{display:flex;gap:10px;align-items:flex-start}.logo{width:38px;height:38px;border-radius:10px;background:#e8f3ff;color:var(--navy);display:grid;place-items:center;font-weight:900;font-size:18px}.brand-title{font-weight:800;font-size:16px;line-height:1.15}.badge{display:inline-block;margin-left:6px;background:#38bdf8;color:#06233a;padding:2px 7px;border-radius:999px;font-size:9px;font-weight:900;vertical-align:2px}.subtitle{font-size:9px;color:#a9bdd1;margin-top:5px;line-height:1.25}.nav{display:flex;justify-content:center;gap:22px}.nav a{color:#b9c8d8;text-decoration:none;font-size:11px;font-weight:700;padding:11px 0;white-space:nowrap}.nav a:hover,.nav a.active{color:white}.nav a.active{border-bottom:2px solid #38bdf8}.header-right{display:flex;align-items:center;gap:14px}.about{font-size:11px;color:#c5d2df;text-decoration:none}.privacy{display:flex;align-items:center;gap:6px;background:rgba(22,163,74,.12);border:1px solid rgba(74,222,128,.25);color:#9af0b2;border-radius:7px;padding:7px 9px;font-size:9px;font-weight:800;white-space:nowrap}.shell{display:grid;grid-template-columns:218px minmax(0,1fr);min-height:calc(100vh - 88px);transition:grid-template-columns .22s ease}.shell.sidebar-collapsed{grid-template-columns:68px minmax(0,1fr)}.sidebar{background:#081d31;border-right:1px solid rgba(255,255,255,.08);padding:12px;position:sticky;top:0;height:calc(100vh - 88px);overflow:auto;transition:padding .22s ease}.sidebar-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}.side-label{font-size:9px;letter-spacing:1.2px;color:#70859a;font-weight:900;padding:0 9px}.sidebar-toggle{width:28px;height:28px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.04);color:#d7e6f4;border-radius:7px;cursor:pointer;font-size:14px;line-height:1}.sidebar-toggle:hover{background:rgba(56,189,248,.12)}.side-link{display:flex;align-items:center;gap:10px;text-decoration:none;color:#a9bbcd;padding:10px;border-radius:8px;margin-bottom:3px;font-size:11px;font-weight:700;white-space:nowrap;overflow:hidden}.side-link:hover,.side-link.active{background:rgba(56,189,248,.1);color:#fff}.side-icon{width:22px;min-width:22px;text-align:center}.side-text{transition:opacity .15s ease}.side-note{margin-top:20px;border:1px solid rgba(56,189,248,.14);background:rgba(56,189,248,.045);border-radius:10px;padding:12px;color:#7f97ad;font-size:9px;line-height:1.5}.shell.sidebar-collapsed .sidebar{padding-left:12px;padding-right:12px}.shell.sidebar-collapsed .sidebar-top{justify-content:center}.shell.sidebar-collapsed .side-label,.shell.sidebar-collapsed .side-note,.shell.sidebar-collapsed .side-text{display:none}.shell.sidebar-collapsed .side-link{justify-content:center;padding-left:0;padding-right:0}.shell.sidebar-collapsed .side-icon{width:24px}.workbench{background:var(--work);color:var(--ink);min-width:0}.content{max-width:1420px;margin:auto;padding:24px 28px 50px}.crumb{font-size:10px;color:#718096;margin-bottom:8px}.page-title{font-size:23px;font-weight:850;letter-spacing:-.5px;margin:0}.page-desc{color:#64748b;font-size:11px;margin:6px 0 22px}.section{margin-top:24px;scroll-margin-top:20px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:11px}.section-title{font-size:12px;letter-spacing:.65px;font-weight:900;color:#26364a}.section-title span{color:#0f766e}.card{background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 12px rgba(20,39,60,.045)}.overview-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{padding:16px}.metric-label{font-size:9px;color:#8290a0;font-weight:900;letter-spacing:.8px}.metric-value{font-size:23px;font-weight:850;margin-top:8px}.metric-sub{font-size:9px;color:#7b8795;margin-top:4px}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.pipe{padding:16px;position:relative}.pipe-no{font-size:9px;color:#2563eb;font-weight:900}.pipe h3{font-size:12px;margin:8px 0 5px}.pipe p{font-size:10px;line-height:1.55;color:#68778a;margin:0}.architecture{padding:22px;overflow:auto}.arch-row{display:flex;align-items:center;justify-content:center;gap:10px;min-width:680px}.arch-box{min-width:145px;text-align:center;padding:14px 12px;border:1px solid #cbd8e6;border-radius:10px;background:#f8fafc}.arch-box strong{display:block;font-size:11px}.arch-box small{font-size:9px;color:#718096}.arrow{font-size:18px;color:#2563eb}.hospitals{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.hospital{padding:15px;text-decoration:none;color:#172033}.hospital:hover{border-color:#60a5fa;box-shadow:0 5px 18px rgba(37,99,235,.1)}.hospital-num{font-size:9px;color:#2563eb;font-weight:900}.hospital h3{font-size:12px;margin:7px 0 2px}.hospital-client{font-size:9px;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:.7px}.hospital-count{font-size:24px;font-weight:900;margin-top:12px;color:#172033}.hospital p{font-size:9px;color:#718096;margin:2px 0 0}.analysis-card{overflow:hidden}.analysis-head{padding:17px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:9px}.check{width:22px;height:22px;border-radius:50%;background:#dcfce7;color:#15803d;display:grid;place-items:center;font-size:12px;font-weight:900}.analysis-body{padding:20px}.status-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;margin-bottom:18px}.status-col{padding:0 22px;border-right:1px solid var(--line)}.status-col:first-child{padding-left:0}.status-col:last-child{border-right:0}.status-label{font-size:9px;color:#8a96a5;letter-spacing:1px;font-weight:900}.prediction{font-size:26px;font-weight:900;margin-top:7px}.descriptor{font-size:10px;color:#6d7b8d;margin-top:3px}.confidence{font-size:30px;font-weight:900;color:var(--green);margin-top:5px}.progress{height:8px;background:#e8edf2;border-radius:20px;overflow:hidden;margin-top:8px;max-width:300px}.progress>i{display:block;height:100%;width:0;background:var(--green);border-radius:20px}.urgency{font-size:18px;font-weight:900;color:var(--red);margin-top:9px}.query{border:1px solid #cfe0f4;background:#f5f9fe;border-radius:9px;padding:13px 15px;margin:18px 0;color:#456b91;font-size:11px}.query-title{font-style:italic;font-weight:800;margin-bottom:5px}.rag-explain{border-top:1px solid var(--line);padding-top:17px}.rag-title{font-size:12px;font-weight:900;display:flex;gap:7px;align-items:center}.doc{color:#2563eb}.rag-explain p,.rag-explain li{font-size:10.5px;line-height:1.65;color:#536275}.rag-explain ul{padding-left:18px}.rag-explain strong{color:#26364a}.guideline{border-top:1px solid var(--line);margin-top:18px;padding-top:15px}.guideline-title{font-size:11.5px;font-weight:900;color:#334155}.guideline p{font-size:11px;line-height:1.6;color:#64748b}.upload-layout{display:block;max-width:380px}.upload-drop{min-height:220px;border:1.5px dashed #b9c8d8;border-radius:10px;background:#f8fafc;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:20px;cursor:pointer}.upload-drop.dragover{border-color:#2563eb;background:#eff6ff}.upload-icon{font-size:35px}.upload-title{font-size:13px;font-weight:850;margin-top:7px}.upload-help{font-size:9px;color:#7a8796;line-height:1.5;margin-top:5px}#xrayFile{display:none}.upload-preview{display:none;margin-top:10px}.upload-preview img{width:100%;height:180px;object-fit:contain;background:#0b1724;border-radius:8px}.upload-actions{display:flex;gap:7px;margin-top:8px}.btn{border:0;border-radius:7px;padding:9px 12px;font-size:10px;font-weight:800;cursor:pointer}.btn-primary{background:#0A2540;color:#fff}.btn-secondary{background:#e8eef4;color:#304258}.btn:disabled{opacity:.45;cursor:not-allowed}.result-panel{border:1px solid var(--line);border-radius:10px;padding:17px;min-height:220px}.pred-row{margin-top:12px}.pred-head{display:flex;justify-content:space-between;font-size:10px}.pred-bar{height:7px;background:#e8edf2;border-radius:20px;margin-top:5px;overflow:hidden}.pred-fill{height:100%;background:#2563eb;border-radius:20px}.rag-box{padding:17px}.rag-input{display:flex;gap:8px}.rag-input input{flex:1;border:1px solid #ccd7e2;border-radius:8px;padding:10px 11px;font-size:11.5px;outline:none}.rag-input input:focus{border-color:#60a5fa}.rag-answer{margin-top:13px;padding:13px;border-radius:8px;background:#f7fafc;border:1px solid var(--line);font-size:10px;line-height:1.6}.sources{margin-top:10px;color:#7b8795;font-size:9px}.foot{text-align:center;color:#8290a0;font-size:9px;margin-top:35px}.detail-link{font-size:9px;color:#2563eb;text-decoration:none;font-weight:800}.modal{display:none;position:fixed;inset:0;background:rgba(2,9,18,.78);align-items:center;justify-content:center;padding:25px;z-index:50}.modal.open{display:flex}.modal-box{max-width:900px;width:100%;background:#fff;border-radius:12px;padding:12px}.modal-top{display:flex;justify-content:space-between;align-items:center;padding:5px 4px 10px;color:#172033;font-size:11px}.close{border:0;background:#e8eef4;color:#304258;border-radius:6px;padding:7px 10px;font-size:9px;font-weight:800;cursor:pointer}.modal-box img{display:block;width:100%;max-height:78vh;object-fit:contain;background:#07111d;border-radius:8px}.sidebar-mobile{display:none}@media(max-width:1100px){.header{grid-template-columns:1fr auto}.nav{display:none}.overview-grid,.pipeline{grid-template-columns:repeat(2,1fr)}.hospitals{grid-template-columns:repeat(3,1fr)}}@media(max-width:780px){.shell{grid-template-columns:1fr}.sidebar{position:relative;height:auto;display:flex;overflow-x:auto;gap:4px;padding:8px}.side-label,.side-note{display:none}.side-link{white-space:nowrap;margin:0}.content{padding:18px 14px 35px}.header{padding:12px 14px}.header-right .about{display:none}.status-grid,.upload-layout{grid-template-columns:1fr}.status-col{border-right:0;border-bottom:1px solid var(--line);padding:12px 0}.status-col:last-child{border-bottom:0}.hospitals{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.overview-grid,.pipeline,.hospitals{grid-template-columns:1fr}.utility{padding:0 10px}.utility-actions{display:none}.brand-title{font-size:14px}.subtitle{font-size:8px}.privacy{display:none}}
</style>
</head>
<body>
<div class="app">
  <div class="utility"><div>Federated Medical RAG Research Workbench</div><div class="utility-actions"><span>Remix</span><span>Share</span><span>Publish</span></div></div>
  <header class="header">
    <a href="/" class="brand" style="text-decoration:none;color:inherit">
      <div class="logo">✚</div><div><div class="brand-title">AI Medical Assistant <span class="badge">FL-RAG</span></div><div class="subtitle">Privacy-Preserving Cardiovascular Clinical Decision Support</div></div>
    </a>
    <nav class="nav"><a href="/" class="active">Home</a><a href="/section/rag">Medical Chatbot</a><a href="/section/image-analysis">Image Analysis</a></nav>
    <div class="header-right"><a class="about" href="/section/about">About</a><div class="privacy">🔒 Zero-PHI Privacy</div></div>
  </header>
  <div class="shell">
    <aside class="sidebar">
      <div class="sidebar-top"><div class="side-label">WORKBENCH</div><button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" aria-label="Collapse sidebar" title="Collapse sidebar">‹</button></div>
      <a class="side-link active" href="/"><span class="side-icon">⌂</span><span class="side-text">System Overview</span></a>
      <a class="side-link" href="/section/workflow"><span class="side-icon">◈</span><span class="side-text">Project Workflow</span></a>
      <a class="side-link" href="/section/architecture"><span class="side-icon">⇄</span><span class="side-text">FL Architecture</span></a>
      <a class="side-link" href="/section/hospitals"><span class="side-icon">▦</span><span class="side-text">Hospital Network</span></a>
      <a class="side-link" href="/section/image-analysis"><span class="side-icon">◉</span><span class="side-text">Image Analysis</span></a>
      <a class="side-link" href="/section/rag"><span class="side-icon">⌁</span><span class="side-text">Medical RAG</span></a>
      <div class="side-note">Clinical research interface<br>Local hospital data • FedAvg • RAG<br>Designed for privacy-preserving analysis.</div>
    </aside>
    <main class="workbench"><div class="content">
      <div class="crumb">WORKBENCH / HOME</div><h1 class="page-title">AI Medical Assistant</h1><p class="page-desc">Privacy-preserving multi-hospital disease detection combining Federated Learning and Retrieval-Augmented Generation.</p>
      <section class="section"><div class="section-head"><div class="section-title">SYSTEM OVERVIEW</div><a class="detail-link" href="/section/overview">Open full page →</a></div><div id="metrics" class="overview-grid"></div></section>
      <section class="section"><div class="section-head"><div class="section-title">HOW THE SYSTEM WORKS</div><a class="detail-link" href="/section/workflow">Open full page →</a></div><div class="pipeline">
        <div class="pipe card"><div class="pipe-no">01</div><h3>Hospital Clients</h3><p>Each hospital keeps medical imaging data locally and acts as a federated client.</p></div>
        <div class="pipe card"><div class="pipe-no">02</div><h3>Federated Learning</h3><p>Local models contribute to a global model through sample-weighted FedAvg.</p></div>
        <div class="pipe card"><div class="pipe-no">03</div><h3>Disease Detection</h3><p>The trained global ResNet18 model predicts six target diseases from chest X-rays.</p></div>
        <div class="pipe card"><div class="pipe-no">04</div><h3>RAG Decision Support</h3><p>Medical literature is retrieved and used to generate patient-friendly explanations.</p></div>
      </div></section>
      <section class="section"><div class="section-head"><div class="section-title">FEDERATED LEARNING ARCHITECTURE</div><a class="detail-link" href="/section/architecture">Open full page →</a></div><div class="architecture card"><div class="arch-row"><div class="arch-box"><strong>Hospital Clients</strong><small>Local X-ray data</small></div><div class="arrow">→</div><div class="arch-box"><strong>Local Models</strong><small>ResNet18</small></div><div class="arrow">→</div><div class="arch-box"><strong>FedAvg Server</strong><small>Sample-weighted aggregation</small></div><div class="arrow">→</div><div class="arch-box"><strong>Global Model</strong><small>Six disease outputs</small></div></div></div></section>
      <section class="section"><div class="section-head"><div class="section-title">FEDERATED HOSPITAL NETWORK</div><a class="detail-link" href="/section/hospitals">Open full page →</a></div><div id="clients" class="hospitals"></div></section>
      <section class="section" id="image-analysis"><div class="section-head"><div class="section-title">IMAGE ANALYSIS</div><a class="detail-link" href="/section/image-analysis">Open full page →</a></div>
        <div class="analysis-card card">
          <div class="analysis-head"><div class="check">✓</div><strong>IMAGE ANALYSIS RESULT</strong></div>
          <div class="analysis-body">
            <div class="status-grid"><div class="status-col"><div class="status-label">PREDICTION</div><div id="mainPrediction" class="prediction">—</div><div id="mainDescriptor" class="descriptor">Upload an X-ray to generate a prediction</div></div><div class="status-col"><div class="status-label">CONFIDENCE SCORE</div><div id="mainConfidence" class="confidence">—</div><div class="progress"><i id="mainProgress" style="width:0%"></i></div></div><div class="status-col"><div class="status-label">CLINICAL URGENCY</div><div id="mainUrgency" class="urgency">—</div><div id="mainUrgencyDescriptor" class="descriptor">Available after prediction</div></div></div>
            <div id="predictionRagQuery" class="query" style="display:none"><div class="query-title">Q: Prediction Converted to Clinical RAG Query:</div><div id="predictionRagQuestion"></div></div>
            
            <div class="guideline"><div class="guideline-title">Clinical Guideline Grounding (2021 ACC/AHA Guideline for the Evaluation and Diagnosis of Chest Pain):</div></div>
          </div>
        </div>
      </section>
      <section class="section"><div class="section-head"><div class="section-title">UPLOAD &amp; RUN GLOBAL FEDAVG MODEL</div><a class="detail-link" href="/section/image-analysis">Open analysis page →</a></div><div class="card" style="padding:18px"><div class="upload-layout"><div><div id="uploadDrop" class="upload-drop" onclick="document.getElementById('xrayFile').click()"><div class="upload-icon">🩻</div><div class="upload-title">Upload Chest X-ray</div><div class="upload-help">Click or drag and drop an image.<br>JPG, JPEG, PNG, WEBP or BMP</div><input id="xrayFile" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp" onchange="handleXraySelected(event)"></div><div id="uploadPreview" class="upload-preview"><img id="xrayPreview" alt="Uploaded chest X-ray preview" onclick="openUploadModal()" title="Click to enlarge"></div><div id="uploadFileName" class="upload-help"></div><div class="upload-actions"><button id="analyzeXrayBtn" class="btn btn-primary" onclick="analyzeUploadedXray()" disabled>Analyze X-ray</button><button class="btn btn-secondary" onclick="clearXrayUpload()">Clear</button></div></div></div></div></section>
      <section class="section"><div class="section-head"><div class="section-title">MEDICAL RAG DECISION SUPPORT</div><a class="detail-link" href="/section/rag">Open chatbot page →</a></div><div class="rag-box card"><div class="rag-input"><input id="ragQuestion" placeholder="Example: What are common clinical features of pneumonia?"><button class="btn btn-primary" onclick="askRAG()">Ask Medical RAG</button></div><div id="ragAnswer" class="rag-answer" style="display:none"></div></div></section>
      <div class="foot">SecureFL • Federated Learning • Differential Privacy • Multi-Disease Detection • Medical RAG</div>
    </div></main>
  </div>
</div>
<div id="uploadModal" class="modal" onclick="closeUploadModal(event)"><div class="modal-box"><div class="modal-top"><b id="uploadModalTitle">Uploaded X-ray</b><button class="close" onclick="closeUploadModal()">Close</button></div><img id="uploadModalImage" alt="Uploaded X-ray enlarged"></div></div>
<script>
const diseases=["Pneumonia","Atelectasis","Cardiomegaly","Effusion","Nodule","Infiltration"];
const fmt=n=>Number(n).toLocaleString("en-IN");
async function get(url){const r=await fetch(url);return await r.json();}
function renderOverview(data){document.getElementById("metrics").innerHTML=[[String(data.hospitals).padStart(2,"0"),"Hospital Clients","Federated sites"],[String(data.diseases).padStart(2,"0"),"Target Diseases","Multi-disease output"],[fmt(data.assigned),"Assigned X-rays","Distributed patient imaging"],[data.aggregation,"Aggregation Method","Global model update"]].map(x=>`<div class="metric card"><div class="metric-label">${x[1]}</div><div class="metric-value">${x[0]}</div><div class="metric-sub">${x[2]}</div></div>`).join("");document.getElementById("clients").innerHTML=data.clients.map(c=>`<a class="hospital card" href="/hospital/${c.id}"><div class="hospital-num">HOSPITAL ${String(c.id).padStart(2,"0")}</div><h3>${c.name||"Hospital "+c.id}</h3><div class="hospital-client">Client ${c.id}</div><div class="hospital-count">${fmt(c.images ?? c.available ?? 0)}</div><p>available X-ray images</p></a>`).join("");}
function toggleSidebar(){const shell=document.querySelector(".shell");const btn=document.getElementById("sidebarToggle");const collapsed=shell.classList.toggle("sidebar-collapsed");btn.textContent=collapsed?"›":"‹";btn.title=collapsed?"Open sidebar":"Collapse sidebar";btn.setAttribute("aria-label",btn.title);localStorage.setItem("sidebarCollapsed",collapsed?"1":"0");}
function restoreSidebar(){const collapsed=localStorage.getItem("sidebarCollapsed")==="1";if(collapsed){document.querySelector(".shell")?.classList.add("sidebar-collapsed");const btn=document.getElementById("sidebarToggle");if(btn){btn.textContent="›";btn.title="Open sidebar";btn.setAttribute("aria-label",btn.title);}}}
async function loadOverview(){restoreSidebar();try{const d=await get("/api/overview");if(d&&d.success!==false)renderOverview(d)}catch(e){console.error(e)}}
let selectedXrayFile=null,previewUrl=null;
function handleXraySelected(e){const f=e.target.files?.[0];if(f)setSelectedXray(f)}
function setSelectedXray(f){selectedXrayFile=f;const url=URL.createObjectURL(f);if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=url;document.getElementById("xrayPreview").src=url;document.getElementById("uploadPreview").style.display="block";document.getElementById("uploadFileName").textContent=f.name;document.getElementById("analyzeXrayBtn").disabled=false;}
function renderUploadedPrediction(data){const rows=(data.diseases||[]).map(d=>{const name=d.disease||d.name||"Unknown";const raw=Number(d.probability??d.confidence??d.score??0);const p=raw<=1?raw*100:raw;return `<div class="pred-row"><div class="pred-head"><span>${name}</span><strong>${p.toFixed(1)}%</strong></div><div class="pred-bar"><div class="pred-fill" style="width:${Math.max(0,Math.min(100,p))}%"></div></div></div>`}).join("");if(document.getElementById("uploadResult")){document.getElementById("uploadResult").innerHTML=`<div style="font-size:11px;font-weight:900">FedAvg ResNet18 • Six-Disease Prediction</div>${rows}<div style="margin-top:12px;font-size:9px;color:#718096">${data.filename||"Uploaded image"} • prediction only; not added to hospital data.</div>`;}const arr=(data.diseases||[]).map(d=>({name:d.disease||d.name,score:(Number(d.probability??d.confidence??d.score??0)<=1?Number(d.probability??d.confidence??d.score??0)*100:Number(d.probability??d.confidence??d.score??0))})).sort((a,b)=>b.score-a.score);if(arr.length){const top=arr[0];document.getElementById("mainPrediction").textContent=top.name;document.getElementById("mainConfidence").textContent=top.score.toFixed(1)+"%";document.getElementById("mainProgress").style.width=Math.max(0,Math.min(100,top.score))+"%";document.getElementById("predictionRagQuery").style.display="block";document.getElementById("predictionRagQuestion").textContent=`What clinical guidance is relevant to a chest X-ray prediction of ${top.name}?`;document.getElementById("mainUrgency").textContent=top.score>=80?"High Priority":top.score>=50?"Review Recommended":"Routine Review";document.getElementById("mainUrgencyDescriptor").textContent=top.score>=80?"Prompt clinical review":top.score>=50?"Review with clinical context":"Consider clinical context";}}
function openUploadModal(){const img=document.getElementById("xrayPreview");if(!img?.src)return;document.getElementById("uploadModalImage").src=img.src;document.getElementById("uploadModal").classList.add("open");}
function closeUploadModal(e){if(e&&e.target!==document.getElementById("uploadModal"))return;document.getElementById("uploadModal").classList.remove("open");document.getElementById("uploadModalImage").src="";}
async function analyzeUploadedXray(){if(!selectedXrayFile)return;const btn=document.getElementById("analyzeXrayBtn");btn.disabled=true;btn.textContent="Analyzing…";if(document.getElementById("uploadResult")){document.getElementById("uploadResult").innerHTML='<div style="font-size:10px;color:#64748b;text-align:center;padding-top:80px">Running the trained global model…</div>';}try{const fd=new FormData();fd.append("image",selectedXrayFile);const r=await fetch("/api/upload-predict",{method:"POST",body:fd});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.error||"Prediction failed");renderUploadedPrediction(d)}catch(e){alert(e.message)}finally{btn.disabled=false;btn.textContent="Analyze X-ray";}}
function clearXrayUpload(){selectedXrayFile=null;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;document.getElementById("xrayFile").value="";document.getElementById("uploadPreview").style.display="none";document.getElementById("uploadFileName").textContent="";document.getElementById("analyzeXrayBtn").disabled=true;if(document.getElementById("uploadResult")){document.getElementById("uploadResult").innerHTML="";}document.getElementById("mainPrediction").textContent="—";document.getElementById("mainDescriptor").textContent="Upload an X-ray to generate a prediction";document.getElementById("mainConfidence").textContent="—";document.getElementById("mainProgress").style.width="0%";document.getElementById("mainUrgency").textContent="—";document.getElementById("mainUrgencyDescriptor").textContent="Available after prediction";document.getElementById("predictionRagQuery").style.display="none";document.getElementById("predictionRagQuestion").textContent="";}
const drop=document.getElementById("uploadDrop");if(drop){["dragenter","dragover"].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.add("dragover")}));["dragleave","drop"].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.remove("dragover")}));drop.addEventListener("drop",e=>{const f=e.dataTransfer.files?.[0];if(f)setSelectedXray(f)})}
async function askRAG(){const q=document.getElementById("ragQuestion").value.trim();if(!q)return;const box=document.getElementById("ragAnswer");box.style.display="block";box.textContent="Retrieving grounded medical information…";try{const r=await fetch("/api/rag",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});const d=await r.json();if(!r.ok)throw new Error(d.error||d.details||"RAG request failed");const sourceNames=(d.sources||[]).map(s=>typeof s==="string"?s:(s.source||s.filename||"Unknown source"));box.innerHTML=`<div>${(d.answer||"No grounded answer returned.").replace(/\n/g,"<br>")}</div>${sourceNames.length?`<div class="sources">Sources: ${sourceNames.join(" • ")}</div>`:""}`}catch(e){box.textContent=e.message}}
loadOverview();
</script>
</body></html>
"""

SECTION_PAGE = r"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>{{ title }} • AI Medical Assistant</title><script src="https://cdn.tailwindcss.com"></script><style>
body{margin:0;background:#061525;color:#e8eef5;font-family:Inter,ui-sans-serif,system-ui}.top{height:36px;background:#061a2c;border-bottom:1px solid #193149;display:flex;justify-content:space-between;align-items:center;padding:0 22px;font-size:11px;color:#9fb2c7}.top span{margin-left:7px;padding:4px 8px;border:1px solid #294057;border-radius:5px}.head{background:#0A2540;padding:14px 22px;display:flex;justify-content:space-between;align-items:center}.brand{display:flex;gap:10px;align-items:center;color:white;text-decoration:none}.logo{width:38px;height:38px;border-radius:10px;background:#e8f3ff;color:#0A2540;display:grid;place-items:center;font-weight:900}.bt{font-weight:800;font-size:16px}.sub{font-size:9px;color:#a9bdd1}.nav{display:flex;gap:18px}.nav a{color:#b9c8d8;text-decoration:none;font-size:10px}.privacy{color:#9af0b2;border:1px solid #275c42;padding:7px;border-radius:7px;font-size:9px}.layout{display:grid;grid-template-columns:218px 1fr;min-height:calc(100vh - 88px);transition:grid-template-columns .22s ease}.layout.sidebar-collapsed{grid-template-columns:68px 1fr}.side{background:#081d31;border-right:1px solid #1a3045;padding:12px;position:sticky;top:0;height:calc(100vh - 88px);overflow:auto}.side-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}.side-label{font-size:9px;letter-spacing:1.2px;color:#70859a;font-weight:900;padding:0 9px}.side-toggle{width:28px;height:28px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.04);color:#d7e6f4;border-radius:7px;cursor:pointer;font-size:14px}.side a{display:flex;align-items:center;gap:10px;color:#a9bbcd;text-decoration:none;padding:10px;border-radius:8px;font-size:11px;font-weight:700;margin:3px 0;white-space:nowrap;overflow:hidden}.side a:hover,.side a.active{background:#11324c;color:white}.side-icon{width:22px;min-width:22px;text-align:center}.side-text{transition:opacity .15s}.layout.sidebar-collapsed .side-top{justify-content:center}.layout.sidebar-collapsed .side-label,.layout.sidebar-collapsed .side-text{display:none}.layout.sidebar-collapsed .side a{justify-content:center}.layout.sidebar-collapsed .side-icon{width:24px}.main{background:#f3f6f9;color:#172033;padding:28px;min-height:100%}.wrap{max-width:1180px;margin:auto}.crumb{font-size:9px;color:#718096}.title{font-size:24px;font-weight:900;margin:6px 0}.desc{font-size:11px;color:#64748b;margin-bottom:18px}.card{background:white;border:1px solid #dbe3ec;border-radius:12px;padding:20px;box-shadow:0 3px 12px rgba(20,39,60,.05)}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.mini{padding:15px}.mini strong{font-size:22px}.mini small{display:block;color:#718096;font-size:9px;margin-top:4px}.pipe{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.pipe h3{font-size:12px}.pipe p{font-size:10px;color:#64748b;line-height:1.55}.arch{display:flex;align-items:center;justify-content:center;gap:10px;min-width:680px}.abox{padding:16px;min-width:145px;text-align:center;background:#f8fafc;border:1px solid #cbd8e6;border-radius:9px}.abox strong{display:block;font-size:11px}.abox small{font-size:9px;color:#718096}.arrow{color:#2563eb}.hosp{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.hosp a{color:#172033;text-decoration:none}.hosp h3{font-size:12px}.hosp p{font-size:9px;color:#718096}.analysis{overflow:hidden;padding:0}.ahead{padding:16px 20px;border-bottom:1px solid #dbe3ec;font-weight:900}.abody{padding:20px}.stats{display:grid;grid-template-columns:1fr 1fr 1fr}.stat{padding:0 20px;border-right:1px solid #dbe3ec}.stat:first-child{padding-left:0}.stat:last-child{border:0}.lab{font-size:9px;color:#8a96a5;letter-spacing:1px;font-weight:900}.pred{font-size:28px;font-weight:900;margin-top:6px}.conf{font-size:32px;color:#16a34a;font-weight:900}.bar{height:8px;background:#e8edf2;border-radius:20px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;width:0;background:#16a34a}.urg{font-size:18px;color:#dc2626;font-weight:900;margin-top:9px}.query{background:#f5f9fe;border:1px solid #cfe0f4;padding:14px;border-radius:8px;margin-top:18px;font-size:10px;color:#456b91}.qtitle{font-style:italic;font-weight:800;margin-bottom:5px}.explain{margin-top:18px;border-top:1px solid #dbe3ec;padding-top:16px;font-size:10px;line-height:1.65;color:#536275}.guideline{border-top:1px solid #dbe3ec;margin-top:15px;padding-top:14px;font-size:11px;color:#64748b;line-height:1.6}.guideline b{font-size:11.5px}.upload{display:block;max-width:380px}.drop{min-height:210px;border:1.5px dashed #b9c8d8;border-radius:10px;background:#f8fafc;display:grid;place-items:center;text-align:center;cursor:pointer;padding:18px}.result{min-height:210px;border:1px solid #dbe3ec;border-radius:10px;padding:16px}.btn{background:#0A2540;color:white;border:0;border-radius:7px;padding:9px 12px;font-size:10px;font-weight:800;cursor:pointer}.rag input{width:calc(100% - 130px);padding:10px;border:1px solid #ccd7e2;border-radius:8px;font-size:11.5px}.rag button{width:110px;margin-left:7px}.modal{display:none;position:fixed;inset:0;background:rgba(2,9,18,.78);align-items:center;justify-content:center;padding:25px;z-index:50}.modal.open{display:flex}.modal-box{max-width:900px;width:100%;background:#fff;border-radius:12px;padding:12px}.modal-top{display:flex;justify-content:space-between;align-items:center;padding:5px 4px 10px;color:#172033;font-size:11px}.close{border:0;background:#e8eef4;color:#304258;border-radius:6px;padding:7px 10px;font-size:9px;font-weight:800;cursor:pointer}.modal-box img{display:block;width:100%;max-height:78vh;object-fit:contain;background:#07111d;border-radius:8px}.foot{margin-top:28px;text-align:center;color:#8290a0;font-size:9px}@media(max-width:900px){.nav{display:none}.layout{grid-template-columns:1fr}.side{display:flex;overflow:auto;gap:3px}.side a{white-space:nowrap}.grid,.pipe{grid-template-columns:1fr 1fr}.hosp{grid-template-columns:1fr 1fr}}@media(max-width:600px){.main{padding:16px}.grid,.pipe,.hosp,.stats,.upload{grid-template-columns:1fr}.stat{border-right:0;border-bottom:1px solid #dbe3ec;padding:12px 0}.stat:last-child{border-bottom:0}}
</style></head><body><div class="top"><div>Federated Medical RAG Research Workbench</div><div><span>Remix</span><span>Share</span><span>Publish</span></div></div><header class="head"><a href="/" class="brand"><div class="logo">✚</div><div><div class="bt">AI Medical Assistant <span style="font-size:9px;background:#38bdf8;color:#06233a;border-radius:99px;padding:2px 6px">FL-RAG</span></div><div class="sub">Privacy-Preserving Cardiovascular Clinical Decision Support</div></div></a><div class="nav"><a href="/">Home</a><a href="/section/rag">Medical Chatbot</a><a href="/section/image-analysis">Image Analysis</a></div><div class="privacy">🔒 Zero-PHI Privacy</div></header><div class="layout"><aside class="side"><div class="side-top"><div class="side-label">WORKBENCH</div><button class="side-toggle" id="sideToggle" onclick="toggleSide()" aria-label="Collapse sidebar" title="Collapse sidebar">‹</button></div><a href="/"><span class="side-icon">⌂</span><span class="side-text">System Overview</span></a><a href="/section/workflow"><span class="side-icon">◈</span><span class="side-text">Project Workflow</span></a><a href="/section/architecture"><span class="side-icon">⇄</span><span class="side-text">FL Architecture</span></a><a class="active" href="/section/hospitals"><span class="side-icon">▦</span><span class="side-text">Hospital Network</span></a><a href="/section/image-analysis"><span class="side-icon">◉</span><span class="side-text">Image Analysis</span></a><a href="/section/rag"><span class="side-icon">⌁</span><span class="side-text">Medical RAG</span></a></aside><main class="main"><div class="wrap"><div class="crumb">WORKBENCH / {{ title|upper }}</div><h1 class="title">{{ title }}</h1><p class="desc">{{ desc }}</p>
{% if section == 'overview' %}<div id="overview" class="grid"></div><div class="foot">Open the sidebar blocks to inspect each part of the research workbench.</div>
{% elif section == 'workflow' %}<div class="pipe"><div class="card"><b>01 • Hospital Clients</b><p>Each hospital keeps medical imaging data locally and acts as a federated client.</p></div><div class="card"><b>02 • Federated Learning</b><p>Local models contribute to a global model through sample-weighted FedAvg.</p></div><div class="card"><b>03 • Disease Detection</b><p>The trained global ResNet18 model predicts six target diseases from chest X-rays.</p></div><div class="card"><b>04 • RAG Decision Support</b><p>Medical literature is retrieved and used to generate patient-friendly explanations.</p></div></div>
{% elif section == 'architecture' %}<div class="card" style="overflow:auto"><div class="arch"><div class="abox"><strong>Hospital Clients</strong><small>Local X-ray data</small></div><div class="arrow">→</div><div class="abox"><strong>Local Models</strong><small>ResNet18</small></div><div class="arrow">→</div><div class="abox"><strong>FedAvg Server</strong><small>Sample-weighted aggregation</small></div><div class="arrow">→</div><div class="abox"><strong>Global Model</strong><small>Six disease outputs</small></div></div></div>
{% elif section == 'hospitals' %}<div id="hospitalList" class="hosp"></div>
{% elif section == 'image-analysis' %}<div class="analysis card"><div class="ahead">✓ &nbsp; IMAGE ANALYSIS RESULT</div><div class="abody"><div class="stats"><div class="stat"><div class="lab">PREDICTION</div><div id="mainPrediction" class="pred">—</div><div>Upload an X-ray to generate a prediction</div></div><div class="stat"><div class="lab">CONFIDENCE SCORE</div><div id="mainConfidence" class="conf">—</div><div class="bar"><i id="mainProgress" style="width:0%"></i></div></div><div class="stat"><div class="lab">CLINICAL URGENCY</div><div id="mainUrgency" class="urg">—</div><div id="mainUrgencyDescriptor">Available after prediction</div></div></div><div id="predictionRagQuery" class="query" style="display:none"><div class="qtitle">Q: Prediction Converted to Clinical RAG Query:</div><span id="predictionRagQuestion"></span><div id="predictionRagAnswer" style="margin-top:10px;color:#536275;line-height:1.65"></div></div><div class="guideline"><b>Clinical Guideline Grounding (2021 ACC/AHA Guideline for the Evaluation and Diagnosis of Chest Pain):</b></div></div></div><div style="height:18px"></div><div class="card"><div class="upload"><div><div id="drop" class="drop" onclick="document.getElementById('file').click()">🩻<br><b>Upload Chest X-ray</b><br><small>JPG, JPEG, PNG, WEBP or BMP</small><input id="file" type="file" accept="image/*" style="display:none" onchange="pick(event)"></div><div id="sectionPreview" style="display:none;margin-top:10px"><img id="sectionPreviewImage" alt="Uploaded X-ray preview" onclick="openSectionModal()" style="width:100%;max-height:220px;object-fit:contain;background:#07111d;border-radius:8px;cursor:zoom-in" title="Click to enlarge"></div><button id="go" class="btn" style="margin-top:8px" disabled onclick="predict()">Analyze X-ray</button></div></div></div><div id="sectionModal" class="modal" onclick="closeSectionModal(event)"><div class="modal-box"><div class="modal-top"><b>Uploaded X-ray</b><button class="close" onclick="closeSectionModal()">Close</button></div><img id="sectionModalImage" alt="Uploaded X-ray enlarged"></div></div>
{% elif section == 'rag' %}<div class="card rag"><input id="q" placeholder="Ask a medical question grounded in the project sources..."><button class="btn" onclick="ask()">Ask Medical RAG</button><div id="ans" style="margin-top:14px;font-size:10px;line-height:1.6"></div></div>
{% elif section == 'about' %}<div class="card"><h3>AI Medical Assistant</h3><p style="font-size:10px;line-height:1.7;color:#64748b">A research workbench combining Federated Learning, privacy-preserving multi-hospital disease detection, and Retrieval-Augmented Generation for evidence-grounded clinical decision support.</p></div>{% endif %}
<div class="foot">SecureFL • Federated Learning • Differential Privacy • Multi-Disease Detection • Medical RAG</div></div></main></div>
<script>
async function get(u){const r=await fetch(u);return await r.json()} let file=null;
function toggleSide(){const layout=document.querySelector(".layout");const btn=document.getElementById("sideToggle");const collapsed=layout.classList.toggle("sidebar-collapsed");btn.textContent=collapsed?"›":"‹";btn.title=collapsed?"Open sidebar":"Collapse sidebar";btn.setAttribute("aria-label",btn.title);localStorage.setItem("sidebarCollapsed",collapsed?"1":"0");}
function restoreSide(){if(localStorage.getItem("sidebarCollapsed")==="1"){document.querySelector(".layout")?.classList.add("sidebar-collapsed");const btn=document.getElementById("sideToggle");if(btn){btn.textContent="›";btn.title="Open sidebar";btn.setAttribute("aria-label",btn.title);}}}

async function init(){restoreSide();try{const d=await get('/api/overview');if(document.getElementById('overview'))document.getElementById('overview').innerHTML=[[d.hospitals,'Hospital Clients'],[d.diseases,'Target Diseases'],[Number(d.assigned).toLocaleString('en-IN'),'Assigned X-rays'],[d.aggregation,'Aggregation Method']].map(x=>`<div class="mini card"><small>${x[1]}</small><strong>${x[0]}</strong></div>`).join('');if(document.getElementById('hospitalList'))document.getElementById('hospitalList').innerHTML=(d.clients||[]).map(c=>`<a class="card" href="/hospital/${c.id}"><div style="font-size:9px;color:#2563eb;font-weight:900;letter-spacing:.8px">HOSPITAL ${String(c.id).padStart(2,'0')}</div><h3>Hospital ${c.id}</h3><div style="font-size:9px;color:#64748b;font-weight:800">Client ${c.id}</div><div style="font-size:24px;font-weight:900;margin-top:12px">${Number(c.images ?? c.available ?? 0).toLocaleString('en-IN')}</div><p>available X-ray images</p></a>`).join('');if(document.getElementById('fl'))document.getElementById('fl').innerHTML=[[d.hospitals,'Hospital Clients'],[d.diseases,'Target Diseases'],[d.aggregation,'Aggregation']].map(x=>`<div class="mini card"><small>${x[1]}</small><strong>${x[0]}</strong></div>`).join('')}catch(e){console.error(e)}}
function pick(e){file=e.target.files?.[0];document.getElementById('go').disabled=!file;if(file){const url=URL.createObjectURL(file);document.getElementById('sectionPreviewImage').src=url;document.getElementById('sectionPreview').style.display='block';}}
function openSectionModal(){const src=document.getElementById('sectionPreviewImage')?.src;if(!src)return;document.getElementById('sectionModalImage').src=src;document.getElementById('sectionModal').classList.add('open')}
function closeSectionModal(e){if(e&&e.target!==document.getElementById('sectionModal'))return;document.getElementById('sectionModal').classList.remove('open');document.getElementById('sectionModalImage').src=''}
async function predict(){if(!file)return;document.getElementById('go').disabled=true;document.getElementById('go').textContent='Analyzing…';try{const fd=new FormData();fd.append('image',file);const r=await fetch('/api/upload-predict',{method:'POST',body:fd});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.error||d.details||'Prediction failed');const a=(d.diseases||[]).map(x=>({n:x.disease||x.name,s:(Number(x.probability??x.confidence??x.score??0)<=1?Number(x.probability??x.confidence??x.score??0)*100:Number(x.probability??x.confidence??x.score??0))})).sort((a,b)=>b.s-a.s);if(a[0]){document.getElementById('mainPrediction').textContent=a[0].n;document.getElementById('mainConfidence').textContent=a[0].s.toFixed(1)+'%';document.getElementById('mainProgress').style.width=Math.max(0,Math.min(100,a[0].s))+'%';document.getElementById('mainUrgency').textContent=a[0].s>=80?'High Priority':a[0].s>=50?'Review Recommended':'Routine Review';document.getElementById('mainUrgencyDescriptor').textContent=a[0].s>=80?'Prompt clinical review':a[0].s>=50?'Review with clinical context':'Consider clinical context';document.getElementById('predictionRagQuery').style.display='block';const q=`What clinical information in the medical sources is relevant to ${a[0].n}, including common chest X-ray findings and clinical features?`;document.getElementById('predictionRagQuestion').textContent=q;const out=document.getElementById('predictionRagAnswer');out.textContent='Retrieving grounded medical information…';try{const rr=await fetch('/api/rag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const rd=await rr.json();if(!rr.ok)throw new Error(rd.error||rd.details||'Clinical RAG request failed');out.innerHTML=(rd.answer||'Information not found in the provided medical sources.').replace(/\n/g,'<br>');}catch(e){out.textContent='Unable to retrieve the grounded clinical information: '+e.message;}}}catch(e){alert(e.message)}finally{document.getElementById('go').disabled=false;document.getElementById('go').textContent='Analyze X-ray'}}

async function ask(){const q=document.getElementById('q').value.trim();if(!q)return;const a=document.getElementById('ans');a.textContent='Retrieving grounded medical information…';try{const r=await fetch('/api/rag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();if(!r.ok)throw new Error(d.error||d.details||'RAG request failed');const sourceNames=(d.sources||[]).map(s=>typeof s==='string'?s:(s.source||s.filename||'Unknown source'));a.innerHTML=`<div>${(d.answer||'No grounded answer returned.').replace(/\n/g,'<br>')}</div>${sourceNames.length?`<div style="margin-top:8px;color:#7b8795;font-size:9px">Sources: ${sourceNames.join(' • ')}</div>`:''}`}catch(e){a.textContent=e.message}}
init();
</script></body></html>
"""

HOSPITAL_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Hospital {{ cid }} • AI Medical Assistant</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
:root{--navy:#0A2540;--navy2:#061a2c;--blue:#2563eb;--cyan:#38bdf8;--ink:#172033;--muted:#64748b;--line:#dbe3ec;--work:#f3f6f9}
*{box-sizing:border-box}body{margin:0;background:#061525;color:#e8eef5;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.utility{height:36px;background:#061a2c;border-bottom:1px solid rgba(255,255,255,.08);display:flex;align-items:center;justify-content:space-between;padding:0 22px;font-size:11px;color:#9fb2c7}.utility-actions{display:flex;gap:8px}.utility-actions span{padding:4px 9px;border:1px solid rgba(255,255,255,.08);border-radius:5px}.header{background:var(--navy);border-bottom:1px solid rgba(255,255,255,.1);padding:14px 22px;display:grid;grid-template-columns:290px 1fr auto;gap:22px;align-items:center}.brand{display:flex;gap:10px;align-items:flex-start;text-decoration:none;color:white}.logo{width:38px;height:38px;border-radius:10px;background:#e8f3ff;color:var(--navy);display:grid;place-items:center;font-weight:900;font-size:18px}.brand-title{font-weight:800;font-size:16px;line-height:1.15}.badge{display:inline-block;margin-left:6px;background:#38bdf8;color:#06233a;padding:2px 7px;border-radius:999px;font-size:9px;font-weight:900;vertical-align:2px}.subtitle{font-size:9px;color:#a9bdd1;margin-top:5px;line-height:1.25}.nav{display:flex;justify-content:center;gap:22px}.nav a{color:#b9c8d8;text-decoration:none;font-size:11px;font-weight:700;padding:11px 0;white-space:nowrap}.nav a:hover{color:white}.header-right{display:flex;align-items:center;gap:14px}.about{font-size:11px;color:#c5d2df;text-decoration:none}.privacy{display:flex;align-items:center;gap:6px;background:rgba(22,163,74,.12);border:1px solid rgba(74,222,128,.25);color:#9af0b2;border-radius:7px;padding:7px 9px;font-size:9px;font-weight:800;white-space:nowrap}.shell{display:grid;grid-template-columns:218px minmax(0,1fr);min-height:calc(100vh - 88px);transition:grid-template-columns .22s ease}.shell.sidebar-collapsed{grid-template-columns:68px minmax(0,1fr)}.sidebar{background:#081d31;border-right:1px solid rgba(255,255,255,.08);padding:12px;position:sticky;top:0;height:calc(100vh - 88px);overflow:auto;transition:padding .22s ease}.sidebar-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}.side-label{font-size:9px;letter-spacing:1.2px;color:#70859a;font-weight:900;padding:0 9px}.sidebar-toggle{width:28px;height:28px;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.04);color:#d7e6f4;border-radius:7px;cursor:pointer;font-size:14px;line-height:1}.sidebar-toggle:hover{background:rgba(56,189,248,.12)}.side-link{display:flex;align-items:center;gap:10px;text-decoration:none;color:#a9bbcd;padding:10px;border-radius:8px;margin-bottom:3px;font-size:11px;font-weight:700;white-space:nowrap;overflow:hidden}.side-link:hover,.side-link.active{background:rgba(56,189,248,.1);color:#fff}.side-icon{width:22px;min-width:22px;text-align:center}.side-text{transition:opacity .15s}.side-note{margin-top:20px;border:1px solid rgba(56,189,248,.14);background:rgba(56,189,248,.045);border-radius:10px;padding:12px;color:#7f97ad;font-size:9px;line-height:1.5}.shell.sidebar-collapsed .sidebar-top{justify-content:center}.shell.sidebar-collapsed .side-label,.shell.sidebar-collapsed .side-note,.shell.sidebar-collapsed .side-text{display:none}.shell.sidebar-collapsed .side-link{justify-content:center}.shell.sidebar-collapsed .side-icon{width:24px}.workbench{background:var(--work);color:var(--ink);min-width:0}.content{max-width:1420px;margin:auto;padding:24px 28px 50px}.crumb{font-size:10px;color:#718096;margin-bottom:8px}.page-title{font-size:23px;font-weight:850;letter-spacing:-.5px;margin:0}.page-desc{color:#64748b;font-size:11px;margin:6px 0 22px}.card{background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 12px rgba(20,39,60,.045)}.back-link{display:inline-flex;align-items:center;gap:6px;color:#2563eb;text-decoration:none;font-size:10px;font-weight:800;margin-bottom:14px}.hero{padding:20px 22px;display:flex;justify-content:space-between;align-items:center;gap:20px}.hero-label{font-size:9px;letter-spacing:1px;color:#2563eb;font-weight:900}.hero h1{margin:6px 0 3px;font-size:24px}.hero p{margin:0;color:#64748b;font-size:10px}.client-pill{background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;border-radius:999px;padding:7px 11px;font-size:9px;font-weight:900;white-space:nowrap}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}.stat{padding:15px}.stat-value{font-size:22px;font-weight:900}.stat-label{font-size:9px;color:#718096;margin-top:4px}.section{margin-top:22px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:10px}.section-title{font-size:12px;letter-spacing:.6px;font-weight:900}.section-desc{font-size:10px;color:#64748b;margin-top:4px}.controls{display:grid;grid-template-columns:1fr 130px;gap:12px;margin-top:12px}.controls label{display:block;font-size:9px;color:#718096;font-weight:800;margin-bottom:5px}.controls select{width:100%;padding:9px;border:1px solid #ccd7e2;border-radius:8px;background:#fff;color:#172033;font-size:10px}.info{font-size:10px;color:#64748b;margin-top:12px}.gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:12px}.image-card{overflow:hidden;border:1px solid var(--line);border-radius:10px;background:#fff;cursor:pointer;transition:.18s}.image-card:hover{transform:translateY(-2px);border-color:#93c5fd;box-shadow:0 5px 18px rgba(37,99,235,.09)}.image-card img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#0b1724}.caption{padding:9px}.image-name{font-size:9px;color:#64748b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tags{color:#2563eb;font-size:8px;margin-top:4px;line-height:1.4}.empty{padding:30px;text-align:center;color:#64748b;font-size:10px}.modal{display:none;position:fixed;inset:0;background:rgba(2,9,18,.78);align-items:center;justify-content:center;padding:25px;z-index:20}.modal.open{display:flex}.modal-box{max-width:850px;width:100%;background:#fff;border-radius:12px;padding:12px}.modal-top{display:flex;justify-content:space-between;align-items:center;padding:5px 4px 10px;color:#172033;font-size:11px}.close{border:0;background:#e8eef4;color:#304258;border-radius:6px;padding:7px 10px;font-size:9px;font-weight:800;cursor:pointer}.modal img{display:block;width:100%;max-height:75vh;object-fit:contain;background:#07111d;border-radius:8px}.foot{text-align:center;color:#8290a0;font-size:9px;margin-top:30px}@media(max-width:1100px){.header{grid-template-columns:1fr auto}.nav{display:none}.gallery{grid-template-columns:repeat(4,1fr)}}@media(max-width:780px){.shell{grid-template-columns:1fr}.sidebar{position:relative;height:auto;display:flex;overflow-x:auto;gap:4px;padding:8px}.sidebar-top{display:none}.side-link{white-space:nowrap;margin:0}.side-label,.side-note{display:none}.content{padding:18px 14px 35px}.header{padding:12px 14px}.header-right .about{display:none}.stats{grid-template-columns:repeat(2,1fr)}.gallery{grid-template-columns:repeat(2,1fr)}.hero{align-items:flex-start;flex-direction:column}.controls{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="utility"><div>Federated Medical RAG Research Workbench</div><div class="utility-actions"><span>Remix</span><span>Share</span><span>Publish</span></div></div>
<header class="header">
  <a class="brand" href="/"><div class="logo">✚</div><div><div class="brand-title">AI Medical Assistant <span class="badge">FL-RAG</span></div><div class="subtitle">Privacy-Preserving Cardiovascular Clinical Decision Support</div></div></a>
  <nav class="nav"><a href="/">Home</a><a href="/section/rag">Medical Chatbot</a><a href="/section/image-analysis">Image Analysis</a></nav>
  <div class="header-right"><a class="about" href="/section/about">About</a><div class="privacy">🔒 Zero-PHI Privacy</div></div>
</header>
<div class="shell">
<aside class="sidebar">
  <div class="sidebar-top"><div class="side-label">WORKBENCH</div><button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()" aria-label="Collapse sidebar" title="Collapse sidebar">‹</button></div>
  <a class="side-link" href="/"><span class="side-icon">⌂</span><span class="side-text">System Overview</span></a>
  <a class="side-link" href="/section/workflow"><span class="side-icon">◈</span><span class="side-text">Project Workflow</span></a>
  <a class="side-link" href="/section/architecture"><span class="side-icon">⇄</span><span class="side-text">FL Architecture</span></a>
  <a class="side-link active" href="/section/hospitals"><span class="side-icon">▦</span><span class="side-text">Hospital Network</span></a>
  <a class="side-link" href="/section/image-analysis"><span class="side-icon">◉</span><span class="side-text">Image Analysis</span></a>
  <a class="side-link" href="/section/rag"><span class="side-icon">⌁</span><span class="side-text">Medical RAG</span></a>
  <div class="side-note">Clinical research interface<br>Local hospital data • FedAvg • RAG<br>Designed for privacy-preserving analysis.</div>
</aside>
<main class="workbench"><div class="content">
  <div class="crumb">WORKBENCH / HOSPITAL NETWORK / CLIENT {{ cid }}</div>
  <a class="back-link" href="/section/hospitals">← Back to Hospital Network</a>
  <div class="hero card"><div><div class="hero-label">FEDERATED HOSPITAL CLIENT</div><h1 id="hospitalName">Hospital {{ "%02d"|format(cid) }}</h1><p>Local X-ray dataset and disease distribution for this federated client.</p></div><div class="client-pill">CLIENT {{ cid }}</div></div>
  <div id="stats" class="stats"></div>
  <section class="section"><div class="section-head"><div><div class="section-title">LOCAL X-RAY DATA</div><div class="section-desc">Browse the images available for this hospital by disease.</div></div></div>
    <div class="card" style="padding:16px"><div class="controls"><div><label>Disease</label><select id="disease" onchange="loadImages()"></select></div><div><label>Images</label><select id="limit" onchange="loadImages()"><option>8</option><option selected>16</option><option>24</option><option>32</option><option>64</option></select></div></div><div id="info" class="info"></div><div id="gallery" class="gallery"></div></div>
  </section>
  <section class="section"><div class="section-head"><div><div class="section-title">DISEASE DISTRIBUTION</div><div class="section-desc">Label counts assigned to this hospital's local dataset.</div></div></div><div id="diseases" class="stats"></div></section>
  <div class="foot">SecureFL • Federated Learning • Differential Privacy • Multi-Disease Detection • Medical RAG</div>
</div></main></div>
<div id="modal" class="modal" onclick="closeModal(event)"><div class="modal-box"><div class="modal-top"><b id="modalTitle">X-ray</b><button class="close" onclick="closeModal()">Close</button></div><img id="modalImage" alt="Selected X-ray"></div></div>
<script>
const cid={{ cid|tojson }};const diseases=["Pneumonia","Atelectasis","Cardiomegaly","Effusion","Nodule","Infiltration"];const fmt=n=>Number(n).toLocaleString("en-IN");
async function get(url){const r=await fetch(url);return await r.json()}
function toggleSidebar(){const shell=document.querySelector(".shell"),btn=document.getElementById("sidebarToggle"),collapsed=shell.classList.toggle("sidebar-collapsed");btn.textContent=collapsed?"›":"‹";btn.title=collapsed?"Open sidebar":"Collapse sidebar";btn.setAttribute("aria-label",btn.title);localStorage.setItem("sidebarCollapsed",collapsed?"1":"0")}
function restoreSidebar(){if(localStorage.getItem("sidebarCollapsed")==="1"){document.querySelector(".shell").classList.add("sidebar-collapsed");const btn=document.getElementById("sidebarToggle");btn.textContent="›";btn.title="Open sidebar";btn.setAttribute("aria-label",btn.title)}}
async function loadHospital(){restoreSidebar();document.getElementById("disease").innerHTML=diseases.map(d=>`<option>${d}</option>`).join("");const d=await get("/api/overview");const h=(d.clients||[]).find(c=>c.id===cid);if(!h){document.getElementById("hospitalName").textContent="Hospital not found";return}document.getElementById("hospitalName").textContent=h.name||`Hospital ${String(cid).padStart(2,"0")}`;document.getElementById("stats").innerHTML=[[fmt(h.images??h.available??0),"Available X-ray Images"],[fmt(h.assigned),"Assigned Records"],[fmt(h.missing),"Missing Images"],["FedAvg","Global Aggregation"]].map(x=>`<div class="stat card"><div class="stat-value">${x[0]}</div><div class="stat-label">${x[1]}</div></div>`).join("");document.getElementById("diseases").innerHTML=diseases.map(x=>`<div class="stat card"><div class="stat-value">${fmt(h.diseases?.[x]||0)}</div><div class="stat-label">${x}</div></div>`).join("");await loadImages()}
async function loadImages(){const disease=document.getElementById("disease").value,limit=document.getElementById("limit").value;const d=await get(`/api/client/${cid}?disease=${encodeURIComponent(disease)}&limit=${limit}`);document.getElementById("info").innerHTML=`Showing <b style="color:#2563eb">${d.images.length}</b> available images for <b style="color:#2563eb">${d.disease}</b> • ${fmt(d.available_matching)} matching images available`;if(!d.images.length){document.getElementById("gallery").innerHTML='<div class="empty" style="grid-column:1/-1">No physical X-ray image is currently available for this disease.</div>';return}document.getElementById("gallery").innerHTML=d.images.map(img=>`<div class="image-card" onclick='openModal(${JSON.stringify(img).replace(/'/g,"&#39;")})'><img src="${img.url}" loading="lazy" alt="${img.name}"><div class="caption"><div class="image-name">${img.name}</div><div class="tags">${img.labels.join(" • ")}</div></div></div>`).join("")}
function openModal(img){document.getElementById("modalTitle").textContent=img.name;document.getElementById("modalImage").src=img.url;document.getElementById("modal").classList.add("open")}
function closeModal(e){if(e&&e.target!==document.getElementById("modal"))return;document.getElementById("modal").classList.remove("open");document.getElementById("modalImage").src=""}
loadHospital();
</script>
</body></html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():

    return render_template_string(
        DASHBOARD_PAGE
    )


@app.get("/section/<section>")
def section_page(section):

    sections = {
        "overview": ("System Overview", "A high-level view of the privacy-preserving medical AI workbench."),
        "workflow": ("Project Workflow", "The complete Hospital → Federated Learning → Disease Detection → RAG workflow."),
        "architecture": ("Federated Learning Architecture", "How local hospital models are combined into the global FedAvg model."),
        "hospitals": ("Hospital Network", "Five simulated hospital clients with dedicated clinical data pages."),
        "image-analysis": ("Image Analysis", "Run the trained global model on a chest X-ray and inspect the clinical result."),
        "rag": ("Medical RAG", "Ask questions and retrieve explanations grounded in the project's medical sources."),
        "about": ("About", "AI Medical Assistant research workbench."),
    }
    if section not in sections:
        abort(404)
    title, desc = sections[section]
    return render_template_string(SECTION_PAGE, section=section, title=title, desc=desc)


@app.get("/hospital/<int:cid>")
def hospital_page(cid):

    if cid not in CLIENTS:
        abort(404)

    return render_template_string(
        HOSPITAL_PAGE,
        cid=cid
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "SecureFL Federated Clinical Intelligence Dashboard"
    )

    print(
        f"Clients loaded: {len(CLIENTS)}"
    )

    print(
        f"Data folder: {CLIENTS_DIR}"
    )

    print(
        "FedAvg prediction:",
        "READY"
        if prediction_error is None
        else "UNAVAILABLE"
    )

    print(
        "RAG:",
        "READY"
        if rag_error is None
        else "UNAVAILABLE"
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 65)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )