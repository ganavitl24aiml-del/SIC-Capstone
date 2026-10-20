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
        predict_xray
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

rag_error = None

try:

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from integration.rag_pipeline import ask_medical_rag

    print("✓ RAG pipeline loaded")

except Exception as e:

    rag_error = str(e)

    print("⚠ RAG pipeline could not be loaded:")
    print(e)


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

        # Load your trained global FedAvg model
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
# API: RAG
# ============================================================

@app.post("/api/rag")
def rag():

    if rag_error:

        return jsonify({

            "error":
                "RAG pipeline is unavailable.",
            "details":
                rag_error

        }), 500

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get("question", "")
    ).strip()

    if not question:

        return jsonify({

            "error":
                "Please enter a question."

        }), 400

    try:

        result = ask_medical_rag(
            question,
            k=5
        )

        return jsonify({

            "question":
                result["question"],

            "answer":
                result["answer"],

            "sources":
                result["sources"]

        })

    except Exception as e:

        return jsonify({

            "error":
                f"RAG request failed: {str(e)}"

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
*{box-sizing:border-box} body{margin:0;background:#061525;color:#e8eef5;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{min-height:100vh;background:#061525}.utility{height:36px;background:#061a2c;border-bottom:1px solid rgba(255,255,255,.08);display:flex;align-items:center;justify-content:space-between;padding:0 22px;font-size:11px;color:#9fb2c7}.utility-actions{display:flex;gap:8px}.utility-actions span{padding:4px 9px;border:1px solid rgba(255,255,255,.08);border-radius:5px}.header{background:var(--navy);border-bottom:1px solid rgba(255,255,255,.1);padding:14px 22px;display:grid;grid-template-columns:290px 1fr auto;gap:22px;align-items:center}.brand{display:flex;gap:10px;align-items:flex-start}.logo{width:38px;height:38px;border-radius:10px;background:#e8f3ff;color:var(--navy);display:grid;place-items:center;font-weight:900;font-size:18px}.brand-title{font-weight:800;font-size:16px;line-height:1.15}.badge{display:inline-block;margin-left:6px;background:#38bdf8;color:#06233a;padding:2px 7px;border-radius:999px;font-size:9px;font-weight:900;vertical-align:2px}.subtitle{font-size:9px;color:#a9bdd1;margin-top:5px;line-height:1.25}.nav{display:flex;justify-content:center;gap:22px}.nav a{color:#b9c8d8;text-decoration:none;font-size:11px;font-weight:700;padding:11px 0;white-space:nowrap}.nav a:hover,.nav a.active{color:white}.nav a.active{border-bottom:2px solid #38bdf8}.header-right{display:flex;align-items:center;gap:14px}.about{font-size:11px;color:#c5d2df;text-decoration:none}.privacy{display:flex;align-items:center;gap:6px;background:rgba(22,163,74,.12);border:1px solid rgba(74,222,128,.25);color:#9af0b2;border-radius:7px;padding:7px 9px;font-size:9px;font-weight:800;white-space:nowrap}.shell{display:grid;grid-template-columns:218px minmax(0,1fr);min-height:calc(100vh - 88px)}.sidebar{background:#081d31;border-right:1px solid rgba(255,255,255,.08);padding:18px 12px;position:sticky;top:0;height:calc(100vh - 88px);overflow:auto}.side-label{font-size:9px;letter-spacing:1.2px;color:#70859a;font-weight:900;padding:0 9px 8px}.side-link{display:flex;align-items:center;gap:10px;text-decoration:none;color:#a9bbcd;padding:10px 10px;border-radius:8px;margin-bottom:3px;font-size:11px;font-weight:700}.side-link:hover,.side-link.active{background:rgba(56,189,248,.1);color:#fff}.side-icon{width:22px;text-align:center}.side-note{margin-top:20px;border:1px solid rgba(56,189,248,.14);background:rgba(56,189,248,.045);border-radius:10px;padding:12px;color:#7f97ad;font-size:9px;line-height:1.5}.workbench{background:var(--work);color:var(--ink);min-width:0}.content{max-width:1420px;margin:auto;padding:24px 28px 50px}.crumb{font-size:10px;color:#718096;margin-bottom:8px}.page-title{font-size:23px;font-weight:850;letter-spacing:-.5px;margin:0}.page-desc{color:#64748b;font-size:11px;margin:6px 0 22px}.section{margin-top:24px;scroll-margin-top:20px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:11px}.section-title{font-size:12px;letter-spacing:.65px;font-weight:900;color:#26364a}.section-title span{color:#0f766e}.card{background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 12px rgba(20,39,60,.045)}.overview-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{padding:16px}.metric-label{font-size:9px;color:#8290a0;font-weight:900;letter-spacing:.8px}.metric-value{font-size:23px;font-weight:850;margin-top:8px}.metric-sub{font-size:9px;color:#7b8795;margin-top:4px}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.pipe{padding:16px;position:relative}.pipe-no{font-size:9px;color:#2563eb;font-weight:900}.pipe h3{font-size:12px;margin:8px 0 5px}.pipe p{font-size:10px;line-height:1.55;color:#68778a;margin:0}.architecture{padding:22px;overflow:auto}.arch-row{display:flex;align-items:center;justify-content:center;gap:10px;min-width:680px}.arch-box{min-width:145px;text-align:center;padding:14px 12px;border:1px solid #cbd8e6;border-radius:10px;background:#f8fafc}.arch-box strong{display:block;font-size:11px}.arch-box small{font-size:9px;color:#718096}.arrow{font-size:18px;color:#2563eb}.hospitals{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.hospital{padding:15px;text-decoration:none;color:#172033}.hospital:hover{border-color:#60a5fa;box-shadow:0 5px 18px rgba(37,99,235,.1)}.hospital-num{font-size:9px;color:#2563eb;font-weight:900}.hospital h3{font-size:12px;margin:7px 0}.hospital p{font-size:9px;color:#718096;margin:0}.analysis-card{overflow:hidden}.analysis-head{padding:17px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:9px}.check{width:22px;height:22px;border-radius:50%;background:#dcfce7;color:#15803d;display:grid;place-items:center;font-size:12px;font-weight:900}.analysis-body{padding:20px}.status-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;margin-bottom:18px}.status-col{padding:0 22px;border-right:1px solid var(--line)}.status-col:first-child{padding-left:0}.status-col:last-child{border-right:0}.status-label{font-size:9px;color:#8a96a5;letter-spacing:1px;font-weight:900}.prediction{font-size:26px;font-weight:900;margin-top:7px}.descriptor{font-size:10px;color:#6d7b8d;margin-top:3px}.confidence{font-size:30px;font-weight:900;color:var(--green);margin-top:5px}.progress{height:8px;background:#e8edf2;border-radius:20px;overflow:hidden;margin-top:8px;max-width:300px}.progress>i{display:block;height:100%;width:88.6%;background:var(--green);border-radius:20px}.urgency{font-size:18px;font-weight:900;color:var(--red);margin-top:9px}.query{border:1px solid #cfe0f4;background:#f5f9fe;border-radius:9px;padding:13px 15px;margin:18px 0;color:#456b91;font-size:11px}.query-title{font-style:italic;font-weight:800;margin-bottom:5px}.rag-explain{border-top:1px solid var(--line);padding-top:17px}.rag-title{font-size:12px;font-weight:900;display:flex;gap:7px;align-items:center}.doc{color:#2563eb}.rag-explain p,.rag-explain li{font-size:10.5px;line-height:1.65;color:#536275}.rag-explain ul{padding-left:18px}.rag-explain strong{color:#26364a}.guideline{border-top:1px solid var(--line);margin-top:18px;padding-top:15px}.guideline-title{font-size:10px;font-weight:900;color:#334155}.guideline p{font-size:10px;line-height:1.6;color:#64748b}.upload-layout{display:grid;grid-template-columns:290px 1fr;gap:18px}.upload-drop{min-height:220px;border:1.5px dashed #b9c8d8;border-radius:10px;background:#f8fafc;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:20px;cursor:pointer}.upload-drop.dragover{border-color:#2563eb;background:#eff6ff}.upload-icon{font-size:35px}.upload-title{font-size:13px;font-weight:850;margin-top:7px}.upload-help{font-size:9px;color:#7a8796;line-height:1.5;margin-top:5px}#xrayFile{display:none}.upload-preview{display:none;margin-top:10px}.upload-preview img{width:100%;height:180px;object-fit:contain;background:#0b1724;border-radius:8px}.upload-actions{display:flex;gap:7px;margin-top:8px}.btn{border:0;border-radius:7px;padding:9px 12px;font-size:10px;font-weight:800;cursor:pointer}.btn-primary{background:#0A2540;color:#fff}.btn-secondary{background:#e8eef4;color:#304258}.btn:disabled{opacity:.45;cursor:not-allowed}.result-panel{border:1px solid var(--line);border-radius:10px;padding:17px;min-height:220px}.pred-row{margin-top:12px}.pred-head{display:flex;justify-content:space-between;font-size:10px}.pred-bar{height:7px;background:#e8edf2;border-radius:20px;margin-top:5px;overflow:hidden}.pred-fill{height:100%;background:#2563eb;border-radius:20px}.rag-box{padding:17px}.rag-input{display:flex;gap:8px}.rag-input input{flex:1;border:1px solid #ccd7e2;border-radius:8px;padding:10px;font-size:10px;outline:none}.rag-input input:focus{border-color:#60a5fa}.rag-answer{margin-top:13px;padding:13px;border-radius:8px;background:#f7fafc;border:1px solid var(--line);font-size:10px;line-height:1.6}.sources{margin-top:10px;color:#7b8795;font-size:9px}.foot{text-align:center;color:#8290a0;font-size:9px;margin-top:35px}.detail-link{font-size:9px;color:#2563eb;text-decoration:none;font-weight:800}.sidebar-mobile{display:none}@media(max-width:1100px){.header{grid-template-columns:1fr auto}.nav{display:none}.overview-grid,.pipeline{grid-template-columns:repeat(2,1fr)}.hospitals{grid-template-columns:repeat(3,1fr)}}@media(max-width:780px){.shell{grid-template-columns:1fr}.sidebar{position:relative;height:auto;display:flex;overflow-x:auto;gap:4px;padding:8px}.side-label,.side-note{display:none}.side-link{white-space:nowrap;margin:0}.content{padding:18px 14px 35px}.header{padding:12px 14px}.header-right .about{display:none}.status-grid,.upload-layout{grid-template-columns:1fr}.status-col{border-right:0;border-bottom:1px solid var(--line);padding:12px 0}.status-col:last-child{border-bottom:0}.hospitals{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.overview-grid,.pipeline,.hospitals{grid-template-columns:1fr}.utility{padding:0 10px}.utility-actions{display:none}.brand-title{font-size:14px}.subtitle{font-size:8px}.privacy{display:none}}
</style>
</head>
<body>
<div class="app">
  <div class="utility"><div>Federated Medical RAG Research Workbench</div><div class="utility-actions"><span>Remix</span><span>Share</span><span>Publish</span></div></div>
  <header class="header">
    <a href="/" class="brand" style="text-decoration:none;color:inherit">
      <div class="logo">✚</div><div><div class="brand-title">AI Medical Assistant <span class="badge">FL-RAG</span></div><div class="subtitle">Privacy-Preserving Cardiovascular Clinical Decision Support</div></div>
    </a>
    <nav class="nav"><a href="/" class="active">Home</a><a href="/section/rag">Medical Chatbot</a><a href="/section/image-analysis">Image Analysis</a><a href="/section/fl-results">Model / FL Results</a><a href="/section/knowledge">Knowledge &amp; Guidelines</a></nav>
    <div class="header-right"><a class="about" href="/section/about">About</a><div class="privacy">🔒 Zero-PHI Privacy</div></div>
  </header>
  <div class="shell">
    <aside class="sidebar">
      <div class="side-label">WORKBENCH</div>
      <a class="side-link active" href="/"><span class="side-icon">⌂</span>System Overview</a>
      <a class="side-link" href="/section/workflow"><span class="side-icon">◈</span>Project Workflow</a>
      <a class="side-link" href="/section/architecture"><span class="side-icon">⇄</span>FL Architecture</a>
      <a class="side-link" href="/section/hospitals"><span class="side-icon">▦</span>Hospital Network</a>
      <a class="side-link" href="/section/image-analysis"><span class="side-icon">◉</span>Image Analysis</a>
      <a class="side-link" href="/section/rag"><span class="side-icon">⌁</span>Medical RAG</a>
      <a class="side-link" href="/section/knowledge"><span class="side-icon">▤</span>Knowledge &amp; Guidelines</a>
      <a class="side-link" href="/section/fl-results"><span class="side-icon">◒</span>Model / FL Results</a>
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
            <div class="status-grid"><div class="status-col"><div class="status-label">PREDICTION</div><div id="mainPrediction" class="prediction">Pneumonia</div><div id="mainDescriptor" class="descriptor">Acute Infectious Consolidation</div></div><div class="status-col"><div class="status-label">CONFIDENCE SCORE</div><div id="mainConfidence" class="confidence">88.6%</div><div class="progress"><i id="mainProgress"></i></div></div><div class="status-col"><div class="status-label">CLINICAL URGENCY</div><div class="urgency">High Priority</div><div class="descriptor">Prompt cardiology evaluation</div></div></div>
            <div class="query"><div class="query-title">Q: Prediction Converted to Clinical RAG Query:</div><div>What are the ATS/IDSA guidelines for community-acquired pneumonia on chest X-ray, air bronchograms, CURB-65 triage, and empiric antibiotics?</div></div>
            <div class="rag-explain"><div class="rag-title"><span class="doc">▤</span>Evidence-Grounded Clinical Explanation (RAG Synthesized):</div><p>The convolutional neural network (ChestXRayVisionNet / 9-Class Dense Softmax) identified radiographic patterns consistent with <strong id="explainDisease">Pneumonia</strong> with a model prediction confidence of <strong id="explainConfidence">88.6%</strong>.</p><ul><li><strong>Radiographic Characterization:</strong> Acute lower respiratory tract infection of the lung parenchyma causing alveolar inflammatory exudation, lobar consolidation with air bronchograms, and silhouette signs.</li><li><strong>Key Radiographic Findings:</strong><ul><li>Homogeneous lobar or segmental air-space consolidation with fluffy margins.</li><li>Branching, dark tubular air bronchograms visible within opacified alveolar spaces.</li><li>Silhouette sign: loss of normal cardiomediastinal or diaphragmatic interface boundaries.</li><li>Patchy bronchopneumonia opacities centered around conducting airways.</li></ul></li></ul></div>
            <div class="guideline"><div class="guideline-title">Clinical Guideline Grounding (2021 ACC/AHA Guideline for the Evaluation and Diagnosis of Chest Pain):</div><p>Chest pain is classified into: (1) Typical Angina meets all 3 criteria: substernal chest discomfort with characteristic quality and duration, provoked by exertion/emotional stress, and relieved by rest/nitroglycerin within minutes. (2) Atypical Angina meets 2 criteria. (3) Non-Anginal meets &lt;=1. (4) Asymptomatic presentation refers to absent subjective chest pain.</p></div>
          </div>
        </div>
      </section>
      <section class="section"><div class="section-head"><div class="section-title">UPLOAD &amp; RUN GLOBAL FEDAVG MODEL</div><a class="detail-link" href="/section/image-analysis">Open analysis page →</a></div><div class="card" style="padding:18px"><div class="upload-layout"><div><div id="uploadDrop" class="upload-drop" onclick="document.getElementById('xrayFile').click()"><div class="upload-icon">🩻</div><div class="upload-title">Upload Chest X-ray</div><div class="upload-help">Click or drag and drop an image.<br>JPG, JPEG, PNG, WEBP or BMP</div><input id="xrayFile" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp" onchange="handleXraySelected(event)"></div><div id="uploadPreview" class="upload-preview"><img id="xrayPreview" alt="Uploaded chest X-ray preview"></div><div id="uploadFileName" class="upload-help"></div><div class="upload-actions"><button id="analyzeXrayBtn" class="btn btn-primary" onclick="analyzeUploadedXray()" disabled>Analyze X-ray</button><button class="btn btn-secondary" onclick="clearXrayUpload()">Clear</button></div></div><div id="uploadResult" class="result-panel"><div style="font-size:10px;color:#7b8795;text-align:center;padding-top:80px">Upload an X-ray and run the global model to see all six disease probabilities.</div></div></div></div></section>
      <section class="section"><div class="section-head"><div class="section-title">MEDICAL RAG DECISION SUPPORT</div><a class="detail-link" href="/section/rag">Open chatbot page →</a></div><div class="rag-box card"><div class="rag-input"><input id="ragQuestion" placeholder="Example: What are common clinical features of pneumonia?"><button class="btn btn-primary" onclick="askRAG()">Ask Medical RAG</button></div><div id="ragAnswer" class="rag-answer" style="display:none"></div></div></section>
      <div class="foot">SecureFL • Federated Learning • Differential Privacy • Multi-Disease Detection • Medical RAG</div>
    </div></main>
  </div>
</div>
<script>
const diseases=["Pneumonia","Atelectasis","Cardiomegaly","Effusion","Nodule","Infiltration"];
const fmt=n=>Number(n).toLocaleString("en-IN");
async function get(url){const r=await fetch(url);return await r.json();}
function renderOverview(data){document.getElementById("metrics").innerHTML=[[String(data.hospitals).padStart(2,"0"),"Hospital Clients","Federated sites"],[String(data.diseases).padStart(2,"0"),"Target Diseases","Multi-disease output"],[fmt(data.assigned),"Assigned X-rays","Distributed patient imaging"],[data.aggregation,"Aggregation Method","Global model update"]].map(x=>`<div class="metric card"><div class="metric-label">${x[1]}</div><div class="metric-value">${x[0]}</div><div class="metric-sub">${x[2]}</div></div>`).join("");document.getElementById("clients").innerHTML=data.clients.map(c=>`<a class="hospital card" href="/hospital/${c.id}"><div class="hospital-num">HOSPITAL ${String(c.id).padStart(2,"0")}</div><h3>${c.name||"Hospital "+c.id}</h3><p>${fmt(c.images||0)} X-rays • Open clinical page →</p></a>`).join("");}
async function loadOverview(){try{const d=await get("/api/overview");if(d&&d.success!==false)renderOverview(d)}catch(e){console.error(e)}}
let selectedXrayFile=null,previewUrl=null;
function handleXraySelected(e){const f=e.target.files?.[0];if(f)setSelectedXray(f)}
function setSelectedXray(f){selectedXrayFile=f;const url=URL.createObjectURL(f);if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=url;document.getElementById("xrayPreview").src=url;document.getElementById("uploadPreview").style.display="block";document.getElementById("uploadFileName").textContent=f.name;document.getElementById("analyzeXrayBtn").disabled=false;}
function renderUploadedPrediction(data){const rows=(data.diseases||[]).map(d=>{const name=d.disease||d.name||"Unknown";const p=Number(d.probability??d.confidence??d.score??0);return `<div class="pred-row"><div class="pred-head"><span>${name}</span><strong>${p.toFixed(1)}%</strong></div><div class="pred-bar"><div class="pred-fill" style="width:${Math.max(0,Math.min(100,p))}%"></div></div></div>`}).join("");document.getElementById("uploadResult").innerHTML=`<div style="font-size:11px;font-weight:900">FedAvg ResNet18 • Six-Disease Prediction</div>${rows}<div style="margin-top:12px;font-size:9px;color:#718096">${data.filename||"Uploaded image"} • prediction only; not added to hospital data.</div>`;const arr=(data.diseases||[]).map(d=>({name:d.disease||d.name,score:Number(d.probability??d.confidence??d.score??0)})).sort((a,b)=>b.score-a.score);if(arr.length){const top=arr[0];document.getElementById("mainPrediction").textContent=top.name;document.getElementById("mainConfidence").textContent=top.score.toFixed(1)+"%";document.getElementById("mainProgress").style.width=Math.max(0,Math.min(100,top.score))+"%";document.getElementById("explainDisease").textContent=top.name;document.getElementById("explainConfidence").textContent=top.score.toFixed(1)+"%";}}
async function analyzeUploadedXray(){if(!selectedXrayFile)return;const btn=document.getElementById("analyzeXrayBtn");btn.disabled=true;btn.textContent="Analyzing…";document.getElementById("uploadResult").innerHTML='<div style="font-size:10px;color:#64748b;text-align:center;padding-top:80px">Running the trained global model…</div>';try{const fd=new FormData();fd.append("image",selectedXrayFile);const r=await fetch("/api/upload-predict",{method:"POST",body:fd});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.error||"Prediction failed");renderUploadedPrediction(d)}catch(e){document.getElementById("uploadResult").innerHTML=`<div style="color:#dc2626;font-size:10px">${e.message}</div>`}finally{btn.disabled=false;btn.textContent="Analyze X-ray";}}
function clearXrayUpload(){selectedXrayFile=null;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;document.getElementById("xrayFile").value="";document.getElementById("uploadPreview").style.display="none";document.getElementById("uploadFileName").textContent="";document.getElementById("analyzeXrayBtn").disabled=true;document.getElementById("uploadResult").innerHTML='<div style="font-size:10px;color:#7b8795;text-align:center;padding-top:80px">Upload an X-ray and run the global model to see all six disease probabilities.</div>'}
const drop=document.getElementById("uploadDrop");if(drop){["dragenter","dragover"].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.add("dragover")}));["dragleave","drop"].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.remove("dragover")}));drop.addEventListener("drop",e=>{const f=e.dataTransfer.files?.[0];if(f)setSelectedXray(f)})}
async function askRAG(){const q=document.getElementById("ragQuestion").value.trim();if(!q)return;const box=document.getElementById("ragAnswer");box.style.display="block";box.textContent="Retrieving grounded medical information…";try{const r=await fetch("/api/rag",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});const d=await r.json();box.innerHTML=`<div>${(d.answer||d.response||"No grounded answer returned.").replace(/\n/g,"<br>")}</div>${d.sources?`<div class="sources">Sources: ${d.sources}</div>`:""}`}catch(e){box.textContent="RAG request failed: "+e.message}}
loadOverview();
</script>
</body></html>
"""

SECTION_PAGE = r"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>{{ title }} • AI Medical Assistant</title><script src="https://cdn.tailwindcss.com"></script><style>
body{margin:0;background:#061525;color:#e8eef5;font-family:Inter,ui-sans-serif,system-ui}.top{height:36px;background:#061a2c;border-bottom:1px solid #193149;display:flex;justify-content:space-between;align-items:center;padding:0 22px;font-size:11px;color:#9fb2c7}.top span{margin-left:7px;padding:4px 8px;border:1px solid #294057;border-radius:5px}.head{background:#0A2540;padding:14px 22px;display:flex;justify-content:space-between;align-items:center}.brand{display:flex;gap:10px;align-items:center;color:white;text-decoration:none}.logo{width:38px;height:38px;border-radius:10px;background:#e8f3ff;color:#0A2540;display:grid;place-items:center;font-weight:900}.bt{font-weight:800;font-size:16px}.sub{font-size:9px;color:#a9bdd1}.nav{display:flex;gap:18px}.nav a{color:#b9c8d8;text-decoration:none;font-size:10px}.privacy{color:#9af0b2;border:1px solid #275c42;padding:7px;border-radius:7px;font-size:9px}.layout{display:grid;grid-template-columns:218px 1fr;min-height:calc(100vh - 88px)}.side{background:#081d31;border-right:1px solid #1a3045;padding:16px 12px}.side a{display:block;color:#a9bbcd;text-decoration:none;padding:10px;border-radius:8px;font-size:11px;font-weight:700;margin:3px 0}.side a:hover,.side a.active{background:#11324c;color:white}.main{background:#f3f6f9;color:#172033;padding:28px;min-height:100%}.wrap{max-width:1180px;margin:auto}.crumb{font-size:9px;color:#718096}.title{font-size:24px;font-weight:900;margin:6px 0}.desc{font-size:11px;color:#64748b;margin-bottom:18px}.card{background:white;border:1px solid #dbe3ec;border-radius:12px;padding:20px;box-shadow:0 3px 12px rgba(20,39,60,.05)}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.mini{padding:15px}.mini strong{font-size:22px}.mini small{display:block;color:#718096;font-size:9px;margin-top:4px}.pipe{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.pipe h3{font-size:12px}.pipe p{font-size:10px;color:#64748b;line-height:1.55}.arch{display:flex;align-items:center;justify-content:center;gap:10px;min-width:680px}.abox{padding:16px;min-width:145px;text-align:center;background:#f8fafc;border:1px solid #cbd8e6;border-radius:9px}.abox strong{display:block;font-size:11px}.abox small{font-size:9px;color:#718096}.arrow{color:#2563eb}.hosp{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.hosp a{color:#172033;text-decoration:none}.hosp h3{font-size:12px}.hosp p{font-size:9px;color:#718096}.analysis{overflow:hidden;padding:0}.ahead{padding:16px 20px;border-bottom:1px solid #dbe3ec;font-weight:900}.abody{padding:20px}.stats{display:grid;grid-template-columns:1fr 1fr 1fr}.stat{padding:0 20px;border-right:1px solid #dbe3ec}.stat:first-child{padding-left:0}.stat:last-child{border:0}.lab{font-size:9px;color:#8a96a5;letter-spacing:1px;font-weight:900}.pred{font-size:28px;font-weight:900;margin-top:6px}.conf{font-size:32px;color:#16a34a;font-weight:900}.bar{height:8px;background:#e8edf2;border-radius:20px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;width:88.6%;background:#16a34a}.urg{font-size:18px;color:#dc2626;font-weight:900;margin-top:9px}.query{background:#f5f9fe;border:1px solid #cfe0f4;padding:14px;border-radius:8px;margin-top:18px;font-size:10px;color:#456b91}.qtitle{font-style:italic;font-weight:800;margin-bottom:5px}.explain{margin-top:18px;border-top:1px solid #dbe3ec;padding-top:16px;font-size:10px;line-height:1.65;color:#536275}.guideline{border-top:1px solid #dbe3ec;margin-top:15px;padding-top:14px;font-size:10px;color:#64748b;line-height:1.6}.upload{display:grid;grid-template-columns:290px 1fr;gap:18px}.drop{min-height:210px;border:1.5px dashed #b9c8d8;border-radius:10px;background:#f8fafc;display:grid;place-items:center;text-align:center;cursor:pointer;padding:18px}.result{min-height:210px;border:1px solid #dbe3ec;border-radius:10px;padding:16px}.btn{background:#0A2540;color:white;border:0;border-radius:7px;padding:9px 12px;font-size:10px;font-weight:800;cursor:pointer}.rag input{width:calc(100% - 130px);padding:10px;border:1px solid #ccd7e2;border-radius:8px}.rag button{width:110px;margin-left:7px}.foot{margin-top:28px;text-align:center;color:#8290a0;font-size:9px}@media(max-width:900px){.nav{display:none}.layout{grid-template-columns:1fr}.side{display:flex;overflow:auto;gap:3px}.side a{white-space:nowrap}.grid,.pipe{grid-template-columns:1fr 1fr}.hosp{grid-template-columns:1fr 1fr}}@media(max-width:600px){.main{padding:16px}.grid,.pipe,.hosp,.stats,.upload{grid-template-columns:1fr}.stat{border-right:0;border-bottom:1px solid #dbe3ec;padding:12px 0}.stat:last-child{border-bottom:0}}
</style></head><body><div class="top"><div>Federated Medical RAG Research Workbench</div><div><span>Remix</span><span>Share</span><span>Publish</span></div></div><header class="head"><a href="/" class="brand"><div class="logo">✚</div><div><div class="bt">AI Medical Assistant <span style="font-size:9px;background:#38bdf8;color:#06233a;border-radius:99px;padding:2px 6px">FL-RAG</span></div><div class="sub">Privacy-Preserving Cardiovascular Clinical Decision Support</div></div></a><div class="nav"><a href="/">Home</a><a href="/section/rag">Medical Chatbot</a><a href="/section/image-analysis">Image Analysis</a><a href="/section/fl-results">Model / FL Results</a><a href="/section/knowledge">Knowledge &amp; Guidelines</a></div><div class="privacy">🔒 Zero-PHI Privacy</div></header><div class="layout"><aside class="side"><a href="/">⌂ System Overview</a><a href="/section/workflow">◈ Project Workflow</a><a href="/section/architecture">⇄ FL Architecture</a><a href="/section/hospitals">▦ Hospital Network</a><a href="/section/image-analysis">◉ Image Analysis</a><a href="/section/rag">⌁ Medical RAG</a><a href="/section/knowledge">▤ Knowledge &amp; Guidelines</a><a href="/section/fl-results">◒ Model / FL Results</a></aside><main class="main"><div class="wrap"><div class="crumb">WORKBENCH / {{ title|upper }}</div><h1 class="title">{{ title }}</h1><p class="desc">{{ desc }}</p>
{% if section == 'overview' %}<div id="overview" class="grid"></div><div class="foot">Open the sidebar blocks to inspect each part of the research workbench.</div>
{% elif section == 'workflow' %}<div class="pipe"><div class="card"><b>01 • Hospital Clients</b><p>Each hospital keeps medical imaging data locally and acts as a federated client.</p></div><div class="card"><b>02 • Federated Learning</b><p>Local models contribute to a global model through sample-weighted FedAvg.</p></div><div class="card"><b>03 • Disease Detection</b><p>The trained global ResNet18 model predicts six target diseases from chest X-rays.</p></div><div class="card"><b>04 • RAG Decision Support</b><p>Medical literature is retrieved and used to generate patient-friendly explanations.</p></div></div>
{% elif section == 'architecture' %}<div class="card" style="overflow:auto"><div class="arch"><div class="abox"><strong>Hospital Clients</strong><small>Local X-ray data</small></div><div class="arrow">→</div><div class="abox"><strong>Local Models</strong><small>ResNet18</small></div><div class="arrow">→</div><div class="abox"><strong>FedAvg Server</strong><small>Sample-weighted aggregation</small></div><div class="arrow">→</div><div class="abox"><strong>Global Model</strong><small>Six disease outputs</small></div></div></div>
{% elif section == 'hospitals' %}<div id="hospitalList" class="hosp"></div>
{% elif section == 'image-analysis' %}<div class="analysis card"><div class="ahead">✓ &nbsp; IMAGE ANALYSIS RESULT</div><div class="abody"><div class="stats"><div class="stat"><div class="lab">PREDICTION</div><div id="mainPrediction" class="pred">Pneumonia</div><div>Acute Infectious Consolidation</div></div><div class="stat"><div class="lab">CONFIDENCE SCORE</div><div id="mainConfidence" class="conf">88.6%</div><div class="bar"><i id="mainProgress"></i></div></div><div class="stat"><div class="lab">CLINICAL URGENCY</div><div class="urg">High Priority</div><div>Prompt cardiology evaluation</div></div></div><div class="query"><div class="qtitle">Q: Prediction Converted to Clinical RAG Query:</div>What are the ATS/IDSA guidelines for community-acquired pneumonia on chest X-ray, air bronchograms, CURB-65 triage, and empiric antibiotics?</div><div class="explain"><b>▤ &nbsp;Evidence-Grounded Clinical Explanation (RAG Synthesized):</b><p>The convolutional neural network (ChestXRayVisionNet / 9-Class Dense Softmax) identified radiographic patterns consistent with <b id="explainDisease">Pneumonia</b> with a model prediction confidence of <b id="explainConfidence">88.6%</b>.</p><ul><li><b>Radiographic Characterization:</b> Acute lower respiratory tract infection of the lung parenchyma causing alveolar inflammatory exudation, lobar consolidation with air bronchograms, and silhouette signs.</li><li><b>Key Radiographic Findings:</b><ul><li>Homogeneous lobar or segmental air-space consolidation with fluffy margins.</li><li>Branching, dark tubular air bronchograms visible within opacified alveolar spaces.</li><li>Silhouette sign: loss of normal cardiomediastinal or diaphragmatic interface boundaries.</li><li>Patchy bronchopneumonia opacities centered around conducting airways.</li></ul></li></ul></div><div class="guideline"><b>Clinical Guideline Grounding (2021 ACC/AHA Guideline for the Evaluation and Diagnosis of Chest Pain):</b><p>Chest pain is classified into: (1) Typical Angina meets all 3 criteria: substernal chest discomfort with characteristic quality and duration, provoked by exertion/emotional stress, and relieved by rest/nitroglycerin within minutes. (2) Atypical Angina meets 2 criteria. (3) Non-Anginal meets &lt;=1. (4) Asymptomatic presentation refers to absent subjective chest pain.</p></div></div></div><div style="height:18px"></div><div class="card"><div class="upload"><div><div id="drop" class="drop" onclick="document.getElementById('file').click()">🩻<br><b>Upload Chest X-ray</b><br><small>JPG, JPEG, PNG, WEBP or BMP</small><input id="file" type="file" accept="image/*" style="display:none" onchange="pick(event)"></div><button id="go" class="btn" style="margin-top:8px" disabled onclick="predict()">Analyze X-ray</button></div><div id="result" class="result">Upload an X-ray to run the trained global FedAvg ResNet18 model and display all six disease probabilities.</div></div></div>
{% elif section == 'rag' %}<div class="card rag"><input id="q" placeholder="Ask a medical question grounded in the project sources..."><button class="btn" onclick="ask()">Ask Medical RAG</button><div id="ans" style="margin-top:14px;font-size:10px;line-height:1.6"></div></div>
{% elif section == 'knowledge' %}<div class="card"><h3>Clinical Knowledge &amp; Guidelines</h3><p style="font-size:10px;line-height:1.7;color:#64748b">This block connects model predictions to the project's retrieval-augmented clinical knowledge layer. The RAG pipeline retrieves relevant medical literature and produces a grounded response instead of relying only on model output.</p><p style="font-size:10px;line-height:1.7;color:#64748b">Use Medical RAG for source-grounded questions and Image Analysis for model predictions. Clinical guidance shown by the application should be treated as decision support, not a substitute for qualified clinical judgment.</p></div>
{% elif section == 'fl-results' %}<div id="fl" class="grid"></div><div class="card" style="margin-top:14px"><h3>Global FedAvg Model</h3><p style="font-size:10px;line-height:1.7;color:#64748b">The current dashboard uses the trained global FedAvg ResNet18 prediction module. Five hospital clients contribute local model information while keeping source imaging local to their client environment.</p></div>
{% elif section == 'about' %}<div class="card"><h3>AI Medical Assistant</h3><p style="font-size:10px;line-height:1.7;color:#64748b">A research workbench combining Federated Learning, privacy-preserving multi-hospital disease detection, and Retrieval-Augmented Generation for evidence-grounded clinical decision support.</p></div>{% endif %}
<div class="foot">SecureFL • Federated Learning • Differential Privacy • Multi-Disease Detection • Medical RAG</div></div></main></div>
<script>
async function get(u){const r=await fetch(u);return await r.json()} let file=null;
async function init(){try{const d=await get('/api/overview');if(document.getElementById('overview'))document.getElementById('overview').innerHTML=[[d.hospitals,'Hospital Clients'],[d.diseases,'Target Diseases'],[Number(d.assigned).toLocaleString('en-IN'),'Assigned X-rays'],[d.aggregation,'Aggregation Method']].map(x=>`<div class="mini card"><small>${x[1]}</small><strong>${x[0]}</strong></div>`).join('');if(document.getElementById('hospitalList'))document.getElementById('hospitalList').innerHTML=(d.clients||[]).map(c=>`<a class="card" href="/hospital/${c.id}"><h3>Hospital ${c.id}</h3><p>${Number(c.images||0).toLocaleString('en-IN')} X-rays • Open dedicated page →</p></a>`).join('');if(document.getElementById('fl'))document.getElementById('fl').innerHTML=[[d.hospitals,'Hospital Clients'],[d.diseases,'Target Diseases'],[d.aggregation,'Aggregation']].map(x=>`<div class="mini card"><small>${x[1]}</small><strong>${x[0]}</strong></div>`).join('')}catch(e){console.error(e)}}
function pick(e){file=e.target.files?.[0];document.getElementById('go').disabled=!file}
async function predict(){if(!file)return;document.getElementById('go').disabled=true;document.getElementById('go').textContent='Analyzing…';const fd=new FormData();fd.append('image',file);try{const r=await fetch('/api/upload-predict',{method:'POST',body:fd});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.error||'Prediction failed');const a=(d.diseases||[]).map(x=>({n:x.disease||x.name,s:Number(x.probability??x.confidence??x.score??0)})).sort((a,b)=>b.s-a.s);document.getElementById('result').innerHTML=a.map(x=>`<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;font-size:10px"><span>${x.n}</span><b>${x.s.toFixed(1)}%</b></div><div class="bar"><i style="width:${x.s}%"></i></div></div>`).join('');if(a[0]){document.getElementById('mainPrediction').textContent=a[0].n;document.getElementById('mainConfidence').textContent=a[0].s.toFixed(1)+'%';document.getElementById('mainProgress').style.width=a[0].s+'%';document.getElementById('explainDisease').textContent=a[0].n;document.getElementById('explainConfidence').textContent=a[0].s.toFixed(1)+'%'}}catch(e){document.getElementById('result').textContent=e.message}finally{document.getElementById('go').disabled=false;document.getElementById('go').textContent='Analyze X-ray'}}
async function ask(){const q=document.getElementById('q').value.trim();if(!q)return;const a=document.getElementById('ans');a.textContent='Retrieving grounded medical information…';try{const r=await fetch('/api/rag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();a.innerHTML=(d.answer||d.response||'No grounded answer returned.').replace(/\n/g,'<br>')}catch(e){a.textContent=e.message}}
init();
</script></body></html>
"""

HOSPITAL_PAGE = r"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>
SecureFL | Hospital
</title>

<style>

* {
    box-sizing:border-box;
}

body {

    margin:0;

    background:#06101d;

    color:#e8eef7;

    font-family:
        Inter,
        Arial,
        sans-serif;
}

main {

    width:min(
        1400px,
        calc(100% - 40px)
    );

    margin:auto;

    padding:
        25px
        0
        70px;
}

.topbar {

    display:flex;

    justify-content:
        space-between;

    align-items:center;

    gap:15px;

    margin-bottom:25px;
}

.back {

    width:auto;

    margin:0;

    padding:
        10px
        16px;
}

.hero {

    padding:30px;

    border-radius:24px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        linear-gradient(
            135deg,
            rgba(20,42,67,.96),
            rgba(8,20,35,.96)
        );
}

.client-label {

    color:#38bdf8;

    font-size:11px;

    font-weight:800;

    letter-spacing:1.5px;
}

h1 {

    margin:
        7px
        0;

    font-size:31px;
}

.subtitle {

    color:#8296ac;

    font-size:13px;
}

.section {

    margin-top:35px;
}

.section h2 {

    margin-bottom:6px;

    font-size:22px;
}

.desc {

    color:#8296ac;

    font-size:12px;
}

.card {

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(13,28,45,.88);

    border-radius:18px;
}

.stats {

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:14px;

    margin-top:18px;
}

.stat {

    padding:20px;
}

.stat-value {

    font-size:25px;

    font-weight:800;
}

.stat-label {

    color:#8296ac;

    font-size:10px;

    text-transform:uppercase;

    margin-top:5px;
}


/* DISEASES */

.diseases {

    display:grid;

    grid-template-columns:
        repeat(6,1fr);

    gap:12px;

    margin-top:18px;
}

.disease {

    padding:17px;
}

.disease-name {

    font-size:11px;

    color:#a9b9c9;
}

.disease-value {

    color:#38bdf8;

    font-size:23px;

    font-weight:800;

    margin-top:7px;
}


/* CONTROLS */

.controls {

    display:grid;

    grid-template-columns:
        1fr 130px;

    gap:12px;

    margin-top:20px;
}

label {

    display:block;

    color:#8296ac;

    font-size:10px;

    margin-bottom:6px;
}

select {

    width:100%;

    padding:12px;

    border-radius:10px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:#0d1b2a;

    color:#e8eef7;
}


/* GALLERY */

.info {

    color:#8296ac;

    font-size:11px;

    margin-top:15px;
}

.gallery {

    display:grid;

    grid-template-columns:
        repeat(5,1fr);

    gap:13px;

    margin-top:17px;
}

.image-card {

    overflow:hidden;

    border-radius:15px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(13,28,45,.88);

    cursor:pointer;

    transition:.2s;
}

.image-card:hover {

    transform:
        translateY(-3px);

    border-color:
        rgba(56,189,248,.4);
}

.image-card img {

    width:100%;

    aspect-ratio:1;

    object-fit:cover;

    display:block;

    background:#020912;
}

.caption {

    padding:10px;
}

.image-name {

    font-size:9px;

    color:#8296ac;

    overflow:hidden;

    text-overflow:ellipsis;

    white-space:nowrap;
}

.tags {

    color:#38bdf8;

    font-size:9px;

    margin-top:5px;

    line-height:1.4;
}


/* PREDICTION */

.prediction {

    margin-top:25px;

    padding:22px;

    display:none;
}

.pred-title {

    font-size:17px;

    font-weight:700;
}

.pred-image {

    color:#8296ac;

    font-size:11px;

    margin-top:4px;
}

.result-row {

    margin-top:15px;
}

.result-head {

    display:flex;

    justify-content:space-between;

    font-size:11px;
}

.bar {

    height:7px;

    background:
        rgba(255,255,255,.06);

    border-radius:20px;

    margin-top:6px;

    overflow:hidden;
}

.fill {

    height:100%;

    background:#38bdf8;

    border-radius:20px;
}


/* RAG */

.rag {

    padding:22px;

    margin-top:18px;
}

.rag-note {

    color:#8296ac;

    font-size:10px;

    margin-top:6px;

    line-height:1.5;
}

.rag-input {

    display:flex;

    gap:10px;

    margin-top:15px;
}

.rag-input input {

    flex:1;

    padding:12px;

    border-radius:10px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:#0d1b2a;

    color:white;

    outline:none;
}

button {

    padding:10px 15px;

    border-radius:10px;

    border:
        1px solid
        rgba(56,189,248,.25);

    background:
        rgba(56,189,248,.08);

    color:#c9eaff;

    font-weight:600;

    cursor:pointer;
}

button:hover {

    background:
        rgba(56,189,248,.16);
}

.rag-answer {

    display:none;

    margin-top:18px;

    padding:17px;

    border-radius:14px;

    border:
        1px solid
        rgba(56,189,248,.12);

    background:
        rgba(56,189,248,.04);

    font-size:12px;

    line-height:1.6;
}

.sources {

    color:#8296ac;

    margin-top:13px;

    font-size:10px;
}

.source {

    margin-top:5px;
}

.empty {

    padding:40px;

    text-align:center;

    color:#8296ac;

    border:
        1px dashed
        rgba(255,255,255,.12);

    border-radius:15px;
}


/* MODAL */

.modal {

    display:none;

    position:fixed;

    inset:0;

    background:
        rgba(0,0,0,.75);

    align-items:center;

    justify-content:center;

    padding:20px;

    z-index:20;
}

.modal-box {

    width:min(
        900px,
        100%
    );

    max-height:90vh;

    overflow:auto;

    padding:22px;

    border-radius:18px;

    background:#091827;

    border:
        1px solid
        rgba(255,255,255,.1);
}

.modal-box img {

    width:100%;

    max-height:500px;

    object-fit:contain;

    background:#020912;

    border-radius:12px;
}

.modal-top {

    display:flex;

    justify-content:space-between;

    align-items:center;

    margin-bottom:15px;
}

.close {

    width:auto;

    margin:0;
}


/* RESPONSIVE */

@media(max-width:1100px) {

    .gallery {

        grid-template-columns:
            repeat(4,1fr);
    }

    .diseases {

        grid-template-columns:
            repeat(3,1fr);
    }
}

@media(max-width:750px) {

    .stats,
    .gallery {

        grid-template-columns:
            repeat(2,1fr);
    }

    .controls {

        grid-template-columns:1fr;
    }

    .rag-input {

        flex-direction:column;
    }
}

@media(max-width:500px) {

    .stats,
    .gallery,
    .diseases {

        grid-template-columns:1fr;
    }
}

</style>

</head>


<body>

<main>


<div class="topbar">

    <button
        class="back"
        onclick="goBack()">

        ← Back to Dashboard

    </button>

</div>


<!-- HOSPITAL HEADER -->

<div class="hero">

    <div class="client-label">
        FEDERATED CLIENT
    </div>

    <h1 id="hospitalName">
        Hospital
    </h1>

    <div class="subtitle">

        This page displays only this hospital's
        federated data and physical X-ray images.

    </div>

</div>


<!-- HOSPITAL STATS -->

<div class="section">

    <h2>
        Hospital Data
    </h2>

    <div class="desc">
        Local client information
    </div>

    <div id="stats"
         class="stats">
    </div>

</div>


<!-- DISEASE DISTRIBUTION -->

<div class="section">

    <h2>
        Disease Distribution
    </h2>

    <div class="desc">
        Disease labels belonging only to this hospital.
    </div>

    <div id="diseases"
         class="diseases">
    </div>

</div>


<!-- X-RAY GALLERY -->

<div class="section">

    <h2>
        X-ray Gallery
    </h2>

    <div class="desc">
        Select a disease to view the corresponding
        X-rays from this hospital.
    </div>


    <div class="controls">

        <div>

            <label>
                Disease
            </label>

            <select id="disease">
            </select>

        </div>


        <div>

            <label>
                Images
            </label>

            <select id="limit">

                <option>8</option>

                <option selected>
                    16
                </option>

                <option>24</option>

                <option>32</option>

            </select>

        </div>

    </div>


    <div id="info"
         class="info">
    </div>


    <div id="gallery"
         class="gallery">
    </div>

</div>


<!-- PREDICTION -->

<div class="section">

    <h2>
        Global Model Prediction
    </h2>

    <div class="desc">
        Select an X-ray above to run the trained
        FedAvg ResNet18 model.
    </div>

    <div id="prediction"
         class="prediction card">
    </div>

</div>


<!-- RAG -->

<div class="section">

    <h2>
        RAG Clinical Decision Support
    </h2>

    <div class="desc">
        Use the medical literature retrieval system
        to understand the predicted disease or ask
        a clinical-information question.
    </div>

    <div class="rag card">

        <div class="rag-note">

            RAG provides source-grounded educational
            information. The X-ray model prediction is
            not a confirmed diagnosis.

        </div>


        <div class="rag-input">

            <input
                id="ragQuestion"
                placeholder="Example: What are common clinical features of pneumonia?"
            >

            <button
                onclick="askRAG()">

                Ask RAG

            </button>

        </div>


        <div id="ragAnswer"
             class="rag-answer">
        </div>

    </div>

</div>


</main>


<!-- IMAGE MODAL -->

<div id="modal"
     class="modal"
     onclick="closeModal(event)">

    <div class="modal-box">

        <div class="modal-top">

            <b id="modalTitle">
                X-ray
            </b>

            <button
                class="close"
                onclick="closeModal()">

                Close

            </button>

        </div>

        <img id="modalImage">

    </div>

</div>


<script>

const cid =
    Number(
        window.location.pathname
        .split("/")
        .pop()
    );


const diseases = [

    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Nodule",
    "Infiltration"

];


const fmt = n =>
    Number(n).toLocaleString("en-IN");


async function get(url) {

    const response =
        await fetch(url);

    return await response.json();
}


/* ============================================================
   LOAD HOSPITAL
   ============================================================ */

async function loadHospital() {

    const diseaseSelect =
        document.getElementById(
            "disease"
        );


    diseaseSelect.innerHTML =
        diseases.map(
            disease =>
            `<option>
                ${disease}
            </option>`
        ).join("");


    const overview =
        await get(
            "/api/overview"
        );


    const hospital =
        overview.clients.find(
            c => c.id === cid
        );


    if (!hospital) {

        document.body.innerHTML =
            "<h2>Hospital not found.</h2>";

        return;
    }


    document.getElementById(
        "hospitalName"
    ).textContent =
        hospital.name;


    document.title =
        `SecureFL | ${hospital.name}`;


    renderStats(hospital);

    renderDiseases(hospital);

    await loadImages();

}


/* ============================================================
   STATS
   ============================================================ */

function renderStats(hospital) {

    document.getElementById(
        "stats"
    ).innerHTML = [

        [
            fmt(hospital.assigned),
            "Assigned X-rays"
        ],

        [
            fmt(hospital.available),
            "Available Images"
        ],

        [
            fmt(hospital.missing),
            "Missing Images"
        ],

        [
            "FedAvg",
            "Global Aggregation"
        ]

    ].map(
        item => `

        <div class="stat card">

            <div class="stat-value">
                ${item[0]}
            </div>

            <div class="stat-label">
                ${item[1]}
            </div>

        </div>

        `
    ).join("");

}


/* ============================================================
   DISEASES
   ============================================================ */

function renderDiseases(hospital) {

    document.getElementById(
        "diseases"
    ).innerHTML =
        diseases.map(
            disease => `

            <div class="disease card">

                <div class="disease-name">
                    ${disease}
                </div>

                <div class="disease-value">
                    ${fmt(
                        hospital.diseases[disease] || 0
                    )}
                </div>

            </div>

            `
        ).join("");

}


/* ============================================================
   LOAD IMAGES
   ============================================================ */

async function loadImages() {

    const disease =
        document.getElementById(
            "disease"
        ).value;

    const limit =
        document.getElementById(
            "limit"
        ).value;


    const data =
        await get(
            `/api/client/${cid}` +
            `?disease=${encodeURIComponent(disease)}` +
            `&limit=${limit}`
        );


    document.getElementById(
        "info"
    ).innerHTML = `

        Showing
        <b style="color:#38bdf8">
            ${data.images.length}
        </b>

        available images for

        <b style="color:#38bdf8">
            ${data.disease}
        </b>

        •
        ${fmt(data.available_matching)}
        matching images available

    `;


    if (!data.images.length) {

        document.getElementById(
            "gallery"
        ).innerHTML = `

            <div class="empty">

                No physical X-ray image is currently
                available for this disease.

            </div>

        `;

        return;
    }


    document.getElementById(
        "gallery"
    ).innerHTML =
        data.images.map(
            image => `

            <div
                class="image-card"
                onclick='selectImage(
                    ${JSON.stringify(image)}
                )'
            >

                <img
                    src="${image.url}"
                    loading="lazy"
                    alt="${image.name}"
                >

                <div class="caption">

                    <div class="image-name">
                        ${image.name}
                    </div>

                    <div class="tags">

                        ${
                            image.labels.join(
                                " • "
                            )
                        }

                    </div>

                </div>

            </div>

            `
        ).join("");

}


/* ============================================================
   SELECT IMAGE
   ============================================================ */

async function selectImage(image) {

    document.getElementById(
        "modalTitle"
    ).textContent =
        image.name;


    document.getElementById(
        "modalImage"
    ).src =
        image.url;


    document.getElementById(
        "modal"
    ).style.display =
        "flex";


    await runPrediction(
        image.name
    );

}


/* ============================================================
   PREDICTION
   ============================================================ */

async function runPrediction(name) {

    const box =
        document.getElementById(
            "prediction"
        );


    box.style.display =
        "block";


    box.innerHTML =
        "Running FedAvg model prediction...";


    box.scrollIntoView({
        behavior:"smooth",
        block:"center"
    });


    try {

        const data =
            await get(
                `/api/predict/${cid}/${encodeURIComponent(name)}`
            );


        if (data.error) {

            box.innerHTML =
                `<b>Prediction error:</b>
                 ${data.error}`;

            return;
        }


        const results =
            data.results || [];


        box.innerHTML = `

            <div class="pred-title">

                FedAvg ResNet18 Prediction

            </div>

            <div class="pred-image">

                ${data.image}

            </div>

            ${results.map(
                result => `

                <div class="result-row">

                    <div class="result-head">

                        <span>
                            ${result.disease}
                        </span>

                        <span>

                            ${
                                (
                                    result.probability
                                    * 100
                                ).toFixed(1)
                            }%

                            ${
                                result.predicted
                                ?
                                " • detected"
                                :
                                ""
                            }

                        </span>

                    </div>

                    <div class="bar">

                        <div
                            class="fill"
                            style="
                                width:${
                                    Math.min(
                                        100,
                                        result.probability * 100
                                    )
                                }%
                            "
                        ></div>

                    </div>

                </div>

                `
            ).join("")}

        `;

    }

    catch(error) {

        box.innerHTML =
            `<b>Prediction error:</b>
             ${error}`;

    }

}


/* ============================================================
   RAG
   ============================================================ */

async function askRAG() {

    const question =
        document.getElementById(
            "ragQuestion"
        ).value.trim();


    const box =
        document.getElementById(
            "ragAnswer"
        );


    if (!question) {

        box.style.display =
            "block";

        box.innerHTML =
            "Please enter a question.";

        return;
    }


    box.style.display =
        "block";

    box.innerHTML =
        "Retrieving medical evidence and generating explanation...";


    try {

        const response =
            await fetch(
                "/api/rag",
                {
                    method:"POST",

                    headers:{
                        "Content-Type":
                            "application/json"
                    },

                    body:JSON.stringify({
                        question:question
                    })
                }
            );


        const data =
            await response.json();


        if (data.error) {

            box.innerHTML =
                `<b>Error:</b>
                 ${data.error}`;

            return;
        }


        const sources =
            (data.sources || [])
            .map(
                source =>
                `<div class="source">
                    • ${source.source}
                </div>`
            )
            .join("");


        box.innerHTML = `

            <div>

                ${escapeHtml(
                    data.answer
                ).replace(
                    /\n/g,
                    "<br>"
                )}

            </div>

            <div class="sources">

                <b>
                    Medical sources used
                </b>

                ${sources}

            </div>

        `;

    }

    catch(error) {

        box.innerHTML =
            `<b>RAG error:</b>
             ${error}`;

    }

}


/* ============================================================
   MODAL
   ============================================================ */

function closeModal(event) {

    if (
        event &&
        event.target !==
        document.getElementById(
            "modal"
        )
    ) {
        return;
    }

    document.getElementById(
        "modal"
    ).style.display =
        "none";
}


function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent =
        text;

    return div.innerHTML;
}


function goBack() {

    window.location.href = "/";

}


/* ============================================================
   EVENTS
   ============================================================ */

document.getElementById(
    "disease"
).onchange =
    loadImages;


document.getElementById(
    "limit"
).onchange =
    loadImages;


/* ============================================================
   START
   ============================================================ */

loadHospital();

</script>

</body>

</html>
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
        "knowledge": ("Knowledge & Guidelines", "Evidence-grounded clinical knowledge connected to the RAG layer."),
        "fl-results": ("Model / FL Results", "Global FedAvg model and federated system status."),
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
        HOSPITAL_PAGE
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