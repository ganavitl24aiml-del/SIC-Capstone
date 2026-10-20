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
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>
SecureFL | Federated Clinical Intelligence
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

body:before {

    content:"";

    position:fixed;

    inset:0;

    pointer-events:none;

    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(30,136,229,.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 5% 90%,
            rgba(0,188,212,.08),
            transparent 25%
        );
}

main {

    width:min(
        1450px,
        calc(100% - 48px)
    );

    margin:auto;

    padding:
        30px
        0
        70px;
}

.card {

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(13,28,45,.88);

    border-radius:20px;
}

.hero {

    padding:34px;

    border-radius:25px;

    background:
        linear-gradient(
            135deg,
            rgba(20,42,67,.96),
            rgba(8,20,35,.96)
        );

    border:
        1px solid
        rgba(255,255,255,.08);
}

.brand {

    font-size:32px;

    font-weight:800;
}

.brand span,
.accent {

    color:#38bdf8;
}

.sub {

    color:#8296ac;

    font-size:14px;

    margin-top:8px;
}

.status {

    display:inline-flex;

    gap:8px;

    align-items:center;

    margin-top:18px;

    padding:
        8px
        13px;

    border-radius:30px;

    color:#86efac;

    background:
        rgba(34,197,94,.09);

    border:
        1px solid
        rgba(34,197,94,.2);

    font-size:11px;

    font-weight:700;
}

.dot {

    width:7px;

    height:7px;

    border-radius:50%;

    background:#22c55e;

    box-shadow:
        0 0 10px
        #22c55e;
}

h2 {

    font-size:23px;

    margin:
        38px
        0
        7px;
}

.desc {

    color:#8296ac;

    font-size:13px;
}

.metrics {

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:16px;

    margin-top:22px;
}

.metric {

    padding:21px;
}

.value {

    font-size:27px;

    font-weight:800;
}

.label {

    color:#8296ac;

    font-size:10px;

    text-transform:uppercase;

    letter-spacing:.8px;

    margin-top:6px;
}


/* PROJECT PIPELINE */

.pipeline {

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:14px;

    margin-top:20px;
}

.pipeline-card {

    padding:20px;

    position:relative;
}

.pipeline-num {

    color:#38bdf8;

    font-size:11px;

    font-weight:800;
}

.pipeline-title {

    font-size:16px;

    font-weight:700;

    margin-top:8px;
}

.pipeline-text {

    color:#8296ac;

    font-size:11px;

    line-height:1.6;

    margin-top:7px;
}


/* HOSPITALS */

.clients {

    display:grid;

    grid-template-columns:
        repeat(5,1fr);

    gap:16px;

    margin-top:22px;
}

.client {

    padding:21px;

    transition:.2s;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(13,28,45,.88);

    border-radius:20px;
}

.client:hover {

    transform:
        translateY(-4px);

    border-color:
        rgba(56,189,248,.45);
}

.num {

    color:#38bdf8;

    font-size:11px;

    font-weight:800;

    letter-spacing:1.4px;
}

.name {

    font-size:20px;

    font-weight:700;

    margin-top:6px;
}

.big {

    font-size:26px;

    font-weight:800;

    margin-top:18px;
}

.muted {

    color:#8296ac;

    font-size:11px;
}

.small {

    margin-top:9px;

    color:#94a7ba;

    font-size:11px;
}

button {

    width:100%;

    margin-top:15px;

    padding:10px;

    border-radius:11px;

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


/* ARCHITECTURE */

.architecture {

    padding:25px;

    margin-top:20px;

    text-align:center;
}

.arch-row {

    display:flex;

    justify-content:center;

    align-items:center;

    gap:10px;

    flex-wrap:wrap;
}

.arch-box {

    padding:
        15px
        20px;

    border-radius:14px;

    border:
        1px solid
        rgba(56,189,248,.2);

    background:
        rgba(56,189,248,.05);

    min-width:150px;
}

.arch-title {

    font-size:13px;

    font-weight:700;
}

.arch-sub {

    font-size:10px;

    color:#8296ac;

    margin-top:5px;
}

.arrow {

    color:#38bdf8;

    font-size:20px;
}


/* RAG */

.rag-box {

    margin-top:20px;

    padding:25px;
}

.rag-input {

    display:flex;

    gap:10px;

    margin-top:16px;
}

.rag-input input {

    flex:1;

    padding:13px;

    border-radius:11px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:#0d1b2a;

    color:white;

    outline:none;
}

.rag-input button {

    width:150px;

    margin-top:0;
}

.rag-answer {

    display:none;

    margin-top:20px;

    padding:18px;

    border-radius:15px;

    background:
        rgba(56,189,248,.04);

    border:
        1px solid
        rgba(56,189,248,.12);

    line-height:1.6;

    font-size:13px;
}

.sources {

    margin-top:15px;

    color:#8296ac;

    font-size:11px;
}

.source {

    margin-top:6px;
}


/* FOOTER */

.foot {

    text-align:center;

    color:#536a80;

    font-size:11px;

    margin-top:55px;
}


/* RESPONSIVE */

@media(max-width:1150px) {

    .clients {

        grid-template-columns:
            repeat(3,1fr);
    }

    .pipeline {

        grid-template-columns:
            repeat(2,1fr);
    }
}

@media(max-width:750px) {

    main {

        width:
            calc(100% - 28px);
    }

    .metrics,
    .clients,
    .pipeline {

        grid-template-columns:
            1fr 1fr;
    }

    .rag-input {

        flex-direction:column;
    }

    .rag-input button {

        width:100%;
    }
}

@media(max-width:500px) {

    .metrics,
    .clients,
    .pipeline {

        grid-template-columns:1fr;
    }

    .arch-row {

        flex-direction:column;
    }

    .arrow {

        transform:rotate(90deg);
    }
}

</style>

</head>


<body>

<main>


<!-- HERO -->

<div class="hero">

    <div class="brand">
        Secure<span>FL</span>
    </div>

    <div class="sub">
        Secure Federated Learning with Differential Privacy
        for Multi-Hospital Disease Detection
    </div>

    <div class="status">
        <span class="dot"></span>
        FEDERATED CLINICAL INTELLIGENCE SYSTEM
    </div>

</div>


<!-- SYSTEM OVERVIEW -->

<h2>System Overview</h2>

<div class="desc">
    Privacy-preserving multi-hospital disease detection
    combining Federated Learning and Retrieval-Augmented
    Generation.
</div>

<div id="metrics"
     class="metrics">
</div>


<!-- PIPELINE -->

<h2>
    How the System Works
</h2>

<div class="desc">
    The dashboard represents the complete project pipeline,
    not only the hospital X-ray data.
</div>

<div class="pipeline">

    <div class="pipeline-card card">

        <div class="pipeline-num">
            01
        </div>

        <div class="pipeline-title">
            Hospital Clients
        </div>

        <div class="pipeline-text">
            Each hospital keeps its medical imaging data
            locally and acts as a federated client.
        </div>

    </div>


    <div class="pipeline-card card">

        <div class="pipeline-num">
            02
        </div>

        <div class="pipeline-title">
            Federated Learning
        </div>

        <div class="pipeline-text">
            Local models contribute to a global model
            through sample-weighted FedAvg.
        </div>

    </div>


    <div class="pipeline-card card">

        <div class="pipeline-num">
            03
        </div>

        <div class="pipeline-title">
            Disease Detection
        </div>

        <div class="pipeline-text">
            The trained global ResNet18 model predicts
            six target diseases from chest X-rays.
        </div>

    </div>


    <div class="pipeline-card card">

        <div class="pipeline-num">
            04
        </div>

        <div class="pipeline-title">
            RAG Decision Support
        </div>

        <div class="pipeline-text">
            Medical literature is retrieved and used to
            generate patient-friendly explanations.
        </div>

    </div>

</div>


<!-- FL ARCHITECTURE -->

<h2>
    Federated Learning Architecture
</h2>

<div class="architecture card">

    <div class="arch-row">

        <div class="arch-box">

            <div class="arch-title">
                Hospital Clients
            </div>

            <div class="arch-sub">
                Local X-ray data
            </div>

        </div>

        <div class="arrow">
            →
        </div>

        <div class="arch-box">

            <div class="arch-title">
                Local Models
            </div>

            <div class="arch-sub">
                ResNet18
            </div>

        </div>

        <div class="arrow">
            →
        </div>

        <div class="arch-box">

            <div class="arch-title">
                FedAvg Server
            </div>

            <div class="arch-sub">
                Sample-weighted aggregation
            </div>

        </div>

        <div class="arrow">
            →
        </div>

        <div class="arch-box">

            <div class="arch-title">
                Global Model
            </div>

            <div class="arch-sub">
                Six disease outputs
            </div>

        </div>

    </div>

</div>


<!-- HOSPITALS -->

<h2>
    Federated Hospital Network
</h2>

<div class="desc">
    Select a hospital to open its dedicated clinical
    data page.
</div>

<div id="clients"
     class="clients">
</div>


<!-- RAG -->

<h2>
    Medical RAG Decision Support
</h2>

<div class="desc">
    Ask a medical question and retrieve an explanation
    grounded in the project's medical sources.
</div>

<div class="rag-box card">

    <div class="rag-input">

        <input
            id="ragQuestion"
            placeholder="Example: What are common clinical features of pneumonia?"
        >

        <button onclick="askRAG()">
            Ask Medical RAG
        </button>

    </div>

    <div id="ragAnswer"
         class="rag-answer">
    </div>

</div>


<div class="foot">

    SecureFL • Federated Learning • Differential Privacy
    • Multi-Disease Detection • Medical RAG

</div>


</main>


<script>

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
   OVERVIEW
   ============================================================ */

function renderOverview(data) {

    document.getElementById(
        "metrics"
    ).innerHTML = [

        [
            String(data.hospitals)
                .padStart(2,"0"),

            "Hospital Clients"
        ],

        [
            String(data.diseases)
                .padStart(2,"0"),

            "Target Diseases"
        ],

        [
            fmt(data.assigned),

            "Assigned X-rays"
        ],

        [
            data.aggregation,

            "Aggregation Method"
        ]

    ].map(item => `

        <div class="metric card">

            <div class="value">
                ${item[0]}
            </div>

            <div class="label">
                ${item[1]}
            </div>

        </div>

    `).join("");


    document.getElementById(
        "clients"
    ).innerHTML = data.clients.map(c => `

        <div class="client">

            <div class="num">
                CLIENT
                ${String(c.id).padStart(2,"0")}
            </div>

            <div class="name">
                ${c.name}
            </div>

            <div class="big">
                ${fmt(c.assigned)}
            </div>

            <div class="muted">
                assigned X-ray images
            </div>

            <div class="small">

                ${fmt(c.available)}
                available

                ${
                    c.missing
                    ?
                    ` • ${fmt(c.missing)} pending`
                    :
                    ""
                }

            </div>

            <button
                onclick="openHospital(${c.id})">

                Open Hospital ${String(c.id)
                    .padStart(2,"0")}

            </button>

        </div>

    `).join("");
}


/* ============================================================
   OPEN HOSPITAL
   ============================================================ */

function openHospital(id) {

    window.location.href =
        `/hospital/${id}`;

}


/* ============================================================
   RAG
   ============================================================ */

async function askRAG() {

    const input =
        document.getElementById(
            "ragQuestion"
        );

    const box =
        document.getElementById(
            "ragAnswer"
        );

    const question =
        input.value.trim();

    if (!question) {

        box.style.display = "block";

        box.innerHTML =
            "Please enter a medical question.";

        return;
    }

    box.style.display = "block";

    box.innerHTML =
        "Generating explanation...";


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
                s =>
                `<div class="source">
                    • ${s.source}
                </div>`
            )
            .join("");


        box.innerHTML = `

            <div>
                ${escapeHtml(
                    data.answer
                ).replace(/\n/g,"<br>")}
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
   SIMPLE HTML ESCAPE
   ============================================================ */

function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent =
        text;

    return div.innerHTML;
}


/* ============================================================
   INITIAL LOAD
   ============================================================ */

(async function() {

    try {

        const data =
            await get(
                "/api/overview"
            );

        renderOverview(data);

    }

    catch(error) {

        console.error(error);

    }

})();

</script>

</body>

</html>
"""


# ============================================================
# HOSPITAL PAGE
# ============================================================

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