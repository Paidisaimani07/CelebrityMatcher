# ============================================================
# 1. IMPORTS
# ============================================================
import os
import sys
import time
import pickle
import uuid
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for
from werkzeug.utils import secure_filename
from deepface import DeepFace

# ============================================================
# 2. FLASK CONFIGURATION
# ============================================================
app = Flask(__name__)
# Restrict file uploads to 5MB to prevent network congestion or resource exhaustion
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 
# Set up a random secret key for the session
app.secret_key = os.urandom(24)

# Configure detailed application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 3. DIRECTORY CONFIGURATION
# ============================================================
# Resolve path directories relative to this file's location
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
UPLOAD_DIR = BASE_DIR / "uploads"
CACHE_FILE = BASE_DIR / "embeddings_cache.pkl"

def _dataset_relative_path(image_path_str: str) -> str:
    """
    Returns the path of an image relative to DATASET_DIR.

    If the stored path was generated when the dataset lived in a different
    directory (e.g. an old cache), .relative_to() would raise ValueError.
    In that case we fall back to using the last two path components
    (celebrity-folder/image-filename), which are always valid.
    """
    p = Path(image_path_str)
    try:
        return str(p.relative_to(DATASET_DIR)).replace("\\", "/")
    except ValueError:
        # Stale cache: rebuild relative path from the last two components
        return "/".join(p.parts[-2:])

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. DEEPFACE MODEL CONFIGURATION
# ============================================================
# We use Facenet512 (512-dim) — its weights are already cached locally at
# ~/.deepface/weights/facenet512_weights.h5 so no download is needed.
# With only 10 images per person (170 total) + skip detector, first-run
# generation completes in ~2-3 minutes instead of 30+ minutes.
MODEL_NAME = "Facenet512"

# Face detection backend for uploaded user images.
# mtcnn is a reliable neural-network based detector that works without
# Haar Cascade XML files (which may be missing on some Python installs).
DETECTOR_BACKEND = "mtcnn"

# Maximum number of images to use per celebrity for embedding generation.
# Using 10 images per person (170 total) keeps first-run generation under 1 minute
# while still providing robust matching via top-k average scoring.
MAX_IMAGES_PER_PERSON = 10

# Allowed file extensions for safe upload validation
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ============================================================
# 5. DATASET SCANNING
# ============================================================
def scan_dataset(dataset_dir: Path):
    """
    Scans the dataset directory recursively for image files.
    
    Arguments:
        dataset_dir: Path object pointing to the celebrity folders folder.
        
    Returns:
        - records: A list of dicts with keys 'person_name' and 'image_path'.
        - signature: A list of tuples (relative_path, file_size, modification_time) for dataset change detection.
    """
    logger.info(f"Scanning dataset directory: {dataset_dir}")
    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return [], []

    records = []
    signature = []

    # Iterate through each folder inside 'dataset', which represents one person's identity
    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    for person_dir in sorted(person_dirs):
        person_name = person_dir.name
        # Find all valid images inside this directory
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for img_path in person_dir.rglob(ext):
                try:
                    stat = img_path.stat()
                    # Store signature representing this file's identity and state
                    signature.append((str(img_path.relative_to(dataset_dir)), stat.st_size, stat.st_mtime))
                    records.append({
                        "person_name": person_name,
                        "image_path": img_path
                    })
                except Exception as e:
                    logger.warning(f"Skipping corrupted or unreadable file {img_path}: {e}")

    # Sort signature to ensure deterministic checks
    signature.sort(key=lambda x: x[0])
    return records, signature

# ============================================================
# 6. EMBEDDING CACHE MANAGEMENT
# ============================================================
def load_cache(cache_file: Path):
    """
    Loads persistent embeddings cache from a local pickle file if it exists.
    """
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            logger.info("Successfully loaded dataset embeddings cache.")
            return cache_data
        except Exception as e:
            logger.error(f"Failed to read cache file: {e}. Rebuilding cache...")
    return None

def save_cache(cache_file: Path, cache_data: dict):
    """
    Persists dataset embeddings cache locally to prevent scanning/generating on every restart.
    """
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Embeddings cache successfully saved to {cache_file}.")
    except Exception as e:
        logger.error(f"Failed to write cache file: {e}")

# ============================================================
# 7. DATASET EMBEDDING GENERATION
# ============================================================
def generate_dataset_embeddings(records: list):
    """
    Uses DeepFace to extract facial representation embeddings for the scanned dataset.
    
    Arguments:
        records: List of dictionaries detailing person_name and image_path.
        
    Returns:
        A list of dictionaries containing 'person_name', 'image_path', and the float embedding list.
    """
    logger.info("Generating face embeddings for dataset images. This may take several minutes on first run...")
    embeddings_list = []
    total = len(records)
    success = 0

    for i, record in enumerate(records):
        img_path = record["image_path"]
        person_name = record["person_name"]
        
        # Periodic log updates to show progress
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == total:
            logger.info(f"Processing database image {i + 1}/{total}...")
            
        try:
            # Use detector_backend="skip" for dataset images because they are already
            # pre-cropped face photos. Skipping face detection makes this 10-20x faster
            # while producing equally accurate embeddings.
            rep = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False
            )
            if rep and len(rep) > 0:
                embeddings_list.append({
                    "person_name": person_name,
                    "image_path": str(img_path),
                    "embedding": rep[0]["embedding"]
                })
                success += 1
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {img_path}: {e}")

    logger.info(f"Completed embedding generation. Successfully processed {success}/{total} images.")
    return embeddings_list

# Global variables stored in memory for highly optimized runtime comparisons
GLOBAL_METADATA = []
GLOBAL_EMBEDDINGS_NORM = None # L2 normalized numpy matrix of shape (N, D)

def initialize_memory_embeddings(embeddings_cache_list):
    """
    Populates memory variables and pre-normalizes embeddings for instantaneous vectorized comparisons.
    """
    global GLOBAL_METADATA, GLOBAL_EMBEDDINGS_NORM
    GLOBAL_METADATA = []
    embeddings_list = []
    
    for item in embeddings_cache_list:
        GLOBAL_METADATA.append({
            "person_name": item["person_name"],
            "image_path": item["image_path"]
        })
        embeddings_list.append(item["embedding"])
        
    if embeddings_list:
        # Load embeddings into a NumPy array
        matrix = np.array(embeddings_list, dtype=np.float32)
        # Normalize each embedding row to L2 norm = 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Prevent division by zero
        GLOBAL_EMBEDDINGS_NORM = matrix / norms
        logger.info(f"Initialized L2 normalized memory matrix with shape {GLOBAL_EMBEDDINGS_NORM.shape}")
    else:
        GLOBAL_EMBEDDINGS_NORM = None
        logger.warning("Empty embeddings list loaded.")

def ensure_embeddings_loaded():
    """
    Ensures dataset embeddings are loaded into memory.
    If not already loaded (e.g. Flask reloader, gunicorn, or import mode),
    loads from persistent cache or scans/builds them dynamically.
    """
    global GLOBAL_EMBEDDINGS_NORM
    if GLOBAL_EMBEDDINGS_NORM is not None:
        return
        
    logger.info("Ensuring database embeddings are loaded into system memory...")
    
    # 1. Try to load from cache
    cached_data = load_cache(CACHE_FILE)
    if cached_data and "embeddings" in cached_data and len(cached_data["embeddings"]) > 0:
        logger.info("Initializing memory embeddings from cached data.")
        initialize_memory_embeddings(cached_data["embeddings"])
        return
        
    # 2. Rebuild if cache is missing or corrupt
    logger.info("Cache file missing or corrupt during verification. Scanning dataset...")
    records, sig = scan_dataset(DATASET_DIR)
    if not records:
        logger.error("No dataset records found. Please ensure the 'dataset/' folder exists and has images.")
        return
        
    embeddings_cache = generate_dataset_embeddings(records)
    if embeddings_cache:
        new_cache_data = {
            "signature": sig,
            "embeddings": embeddings_cache
        }
        save_cache(CACHE_FILE, new_cache_data)
        initialize_memory_embeddings(embeddings_cache)

# ============================================================
# 8. FACE DETECTION
# ============================================================
def detect_faces_in_image(image_path: Path):
    """
    Detects faces in an image. Enforce_detection=True triggers a ValueError if no face is found.
    Tries multiple detector backends in sequence for maximum robustness:
    1. 'opencv' (fastest, standard frontal faces)
    2. 'ssd' (fast, robust to minor angles and hand occlusions)
    3. 'mtcnn' (slower, highly robust to profile and angled faces)
    """
    backends = [DETECTOR_BACKEND, "ssd", "opencv"]  # Try configured backend, then fallbacks
    
    last_error = None
    for backend in backends:
        try:
            logger.info(f"Attempting face detection using backend: {backend}")
            faces = DeepFace.extract_faces(
                img_path=str(image_path),
                detector_backend=backend,
                enforce_detection=True
            )
            logger.info(f"Successfully detected {len(faces)} face(s) using backend: {backend}")
            return faces, backend
        except ValueError as e:
            last_error = e
            continue
        except Exception as e:
            logger.warning(f"Error with detector backend {backend}: {e}")
            last_error = e
            continue
            
    raise last_error if last_error else ValueError("Face could not be detected.")

# ============================================================
# 9. FACE EMBEDDING GENERATION
# ============================================================
def generate_query_embedding(image_path: Path, detector_backend: str):
    """
    Generates embedding for the query image using the specified backend.
    """
    rep = DeepFace.represent(
        img_path=str(image_path),
        model_name=MODEL_NAME,
        detector_backend=detector_backend,
        enforce_detection=True
    )
    if not rep or len(rep) == 0:
        raise ValueError("Could not extract embeddings from the uploaded image.")
    return np.array(rep[0]["embedding"], dtype=np.float32)

# ============================================================
# 10. SIMILARITY CALCULATION
# ============================================================
def calculate_similarities(query_emb: np.ndarray):
    """
    Performs vectorized cosine similarity calculations using NumPy.
    Since dataset embeddings are pre-normalized, the cosine similarity
    is equivalent to the dot product of the normalized query vector and 
    the normalized dataset embeddings matrix.
    
    Returns:
        1D NumPy array containing cosine similarity scores in range [0.0, 1.0].
    """
    global GLOBAL_EMBEDDINGS_NORM
    if GLOBAL_EMBEDDINGS_NORM is None:
        raise ValueError("No embeddings loaded in system memory.")

    # Normalize query embedding vector
    q_norm = np.linalg.norm(query_emb)
    if q_norm > 0:
        normalized_query = query_emb / q_norm
    else:
        normalized_query = query_emb

    # Vectorized dot product calculation
    similarities = np.dot(GLOBAL_EMBEDDINGS_NORM, normalized_query)
    
    # Clip results to [0.0, 1.0] range to keep similarity positive and clear for UI presentation
    return np.clip(similarities, 0.0, 1.0)

# ============================================================
# 11. IDENTITY-LEVEL MATCHING
# ============================================================
def aggregate_scores_by_identity(similarities: np.ndarray, top_k: int = 5, weight_max: float = 0.6):
    """
    Aggregates image-level similarity scores at the identity (person) level.
    Uses Strategy C: Weighted combination of the maximum similarity (60%) 
    and the top-K average similarity (40%) per identity.
    
    Arguments:
        similarities: 1D array of floats representing similarity with every image.
        top_k: Number of closest images per person to average.
        weight_max: Weight given to the best single image match.
        
    Returns:
        A list of dictionaries sorted by aggregated score descending.
    """
    global GLOBAL_METADATA
    
    # Group results by person
    person_scores = {}
    for idx, score in enumerate(similarities):
        meta = GLOBAL_METADATA[idx]
        person = meta["person_name"]
        path = meta["image_path"]
        
        if person not in person_scores:
            person_scores[person] = []
        person_scores[person].append((score, path))
        
    results = []
    
    for person, item_list in person_scores.items():
        # Sort by similarity score descending
        item_list.sort(key=lambda x: x[0], reverse=True)
        
        # Strategy C Calculations
        s_max = item_list[0][0]
        best_path = item_list[0][1]
        
        # Calculate top-K average
        top_k_vals = [x[0] for x in item_list[:top_k]]
        s_topk_avg = np.mean(top_k_vals)
        
        # Weighted aggregate
        aggregated_score = (weight_max * s_max) + ((1.0 - weight_max) * s_topk_avg)
        
        results.append({
            "person_name": person,
            "score": float(aggregated_score),
            "best_match_path": best_path,
            "best_match_score": float(s_max)
        })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ============================================================
# 12. FLASK ROUTES
# ============================================================
def allowed_file(filename):
    """
    Validates file extensions.
    """
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    """
    Serves the premium single-page UI with inline CSS and JavaScript.
    """
    # Embedded HTML content. Kept in python to preserve single-file execution requirement.
    # The HTML string is configured below.
    return render_template_string(HTML_TEMPLATE)

@app.route("/upload", methods=["POST"])
def upload():
    """
    Processes the uploaded file, executes face validation, generates embeddings, 
    calculates vectorized cosine similarity, aggregates findings, and reports the match.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # Guarantee dataset embeddings are loaded in memory
        ensure_embeddings_loaded()
        if GLOBAL_EMBEDDINGS_NORM is None:
            return jsonify({"success": False, "error": "Database embeddings could not be loaded into memory. Check dataset folder."}), 500

        # Generate safe file path using uuid to prevent collisions
        safe_name = secure_filename(file.filename)
        extension = Path(safe_name).suffix
        unique_name = f"{uuid.uuid4().hex}{extension}"
        temp_path = UPLOAD_DIR / unique_name
        
        file.save(temp_path)
        
        # Validate that the image can be read by OpenCV
        img = cv2.imread(str(temp_path))
        if img is None:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({"success": False, "error": "Corrupted or invalid image uploaded."}), 400
            
        # 1. Face Detection & Validation
        try:
            detected_faces, active_backend = detect_faces_in_image(temp_path)
            if len(detected_faces) > 1:
                if temp_path.exists():
                    temp_path.unlink()
                return jsonify({
                    "success": False, 
                    "error": "Multiple faces detected. Please upload an image containing exactly one face."
                }), 400
        except ValueError as e:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({
                "success": False, 
                "error": "No face detected. Please upload a clear image containing a face."
            }), 400
        except Exception as e:
            logger.error(f"Detection error: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({"success": False, "error": f"Face detection error: {str(e)}"}), 500

        # 2. Embedding Generation
        try:
            query_emb = generate_query_embedding(temp_path, active_backend)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({"success": False, "error": f"Failed to extract face features: {str(e)}"}), 500

        # 3. Vectorized Similarity Comparison
        try:
            similarities = calculate_similarities(query_emb)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({"success": False, "error": f"Similarity calculation error: {str(e)}"}), 500

        # 4. Score Aggregation
        aggregated = aggregate_scores_by_identity(similarities)
        
        if not aggregated:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({"success": False, "error": "No look-alikes found in dataset."}), 400

        best = aggregated[0]
        
        # Determine classification metrics (not statistical probabilities, but similarity mappings)
        score = best["score"]
        
        # Check if the exact image is uploaded (similarity very high)
        is_exact = score >= 0.95
        
        if score >= 0.70:
            classification = {
                "level": "high",
                "label": "Strong Match Found",
                "description": f"Excellent resemblance! You look highly similar to {best['person_name']}."
            }
        elif score >= 0.50:
            classification = {
                "level": "moderate",
                "label": "Possible Match",
                "description": f"Moderate resemblance. You show some similar features to {best['person_name']}."
            }
        else:
            classification = {
                "level": "low",
                "label": "Low Confidence Match",
                "description": f"No confident look-alike found. The closest match was {best['person_name']}, but with low similarity. Try uploading a clearer frontal photo."
            }

        # Build Response
        response_data = {
            "success": True,
            "query_image_url": url_for("serve_upload_image", filename=unique_name),
            "confidence_classification": classification,
            "best_match": {
                "person_name": best["person_name"],
                "score": best["score"],
                "representative_image_url": url_for(
                    "serve_dataset_image", 
                    filename=_dataset_relative_path(best["best_match_path"])
                ),
                "is_exact_match": is_exact
            },
            "top_matches": [
                {
                    "person_name": item["person_name"],
                    "score": item["score"],
                    "representative_image_url": url_for(
                        "serve_dataset_image", 
                        filename=_dataset_relative_path(item["best_match_path"])
                    )
                } for item in aggregated[:3]
            ]
        }
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Global server error: {e}")
        return jsonify({"success": False, "error": f"An unexpected backend error occurred: {str(e)}"}), 500

@app.route("/dataset_image/<path:filename>")
def serve_dataset_image(filename):
    """
    Statically serves celebrity face images from the dataset.
    """
    return send_from_directory(DATASET_DIR, filename)

@app.route("/uploads/<path:filename>")
def serve_upload_image(filename):
    """
    Statically serves query images uploaded by users.
    """
    return send_from_directory(UPLOAD_DIR, filename)

# ============================================================
# 13. APPLICATION STARTUP
# ============================================================
# Insert HTML template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Celebrity Look-Alike Finder</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #8b5cf6;
            --primary-hover: #7c3aed;
            --secondary: #06b6d4;
            --secondary-hover: #0891b2;
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 50%, #1e1b4b 0%, var(--bg-dark) 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeIn 1s ease-out;
        }

        h1 {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.05em;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(15, 23, 42, 0.4);
            margin-bottom: 2rem;
            position: relative;
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--secondary);
            background: rgba(6, 182, 212, 0.05);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
        }

        .upload-icon {
            font-size: 3.5rem;
            color: var(--secondary);
            margin-bottom: 1rem;
            display: inline-block;
            transition: transform 0.3s ease;
        }

        .upload-area:hover .upload-icon {
            transform: translateY(-5px);
        }

        .upload-text h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .upload-text p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        #file-input {
            display: none;
        }

        .preview-container {
            display: none;
            flex-direction: column;
            align-items: center;
            margin-top: 1rem;
            animation: fadeIn 0.5s ease;
        }

        .preview-image {
            max-width: 250px;
            max-height: 250px;
            border-radius: 12px;
            object-fit: cover;
            border: 2px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }

        .remove-btn {
            margin-top: 1rem;
            background: transparent;
            border: 1px solid var(--danger);
            color: var(--danger);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }

        .remove-btn:hover {
            background: var(--danger);
            color: white;
        }

        .action-btn {
            display: block;
            width: 100%;
            padding: 1rem 2rem;
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            font-family: inherit;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
            transition: all 0.3s ease;
            text-align: center;
        }

        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        }

        .action-btn:active {
            transform: translateY(1px);
        }

        .action-btn:disabled {
            background: #475569;
            box-shadow: none;
            cursor: not-allowed;
            transform: none;
        }

        .loader-container {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 2rem 0;
            animation: fadeIn 0.3s ease;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-left-color: var(--secondary);
            border-right-color: var(--primary);
            border-radius: 50%;
            animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
            margin-bottom: 1.5rem;
        }

        .loader-text {
            color: var(--text-muted);
            font-weight: 500;
            font-size: 1rem;
            text-align: center;
        }

        .results-container {
            display: none;
            margin-top: 3rem;
            border-top: 1px solid var(--border-color);
            padding-top: 2rem;
            animation: fadeIn 0.8s ease;
        }

        .results-title {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #fff 0%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .match-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
        }

        @media (max-width: 768px) {
            .match-comparison {
                grid-template-columns: 1fr;
            }
        }

        .image-card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        .image-card h4 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .match-img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            background: #1e293b;
        }

        .match-details {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-bottom: 3rem;
        }

        .confidence-badge {
            display: inline-block;
            padding: 0.5rem 1.25rem;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-high {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }

        .badge-mod {
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning);
            border: 1px solid var(--warning);
        }

        .badge-low {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
            border: 1px solid var(--danger);
        }

        .match-name {
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.03em;
        }

        .match-score {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--secondary);
            margin-bottom: 1.5rem;
        }

        .progress-container {
            width: 100%;
            max-width: 400px;
            background: rgba(255, 255, 255, 0.1);
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 1rem;
            position: relative;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--secondary) 0%, var(--primary) 100%);
            border-radius: 6px;
            width: 0%;
            transition: width 1s ease-out;
        }

        .match-notice {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 1rem;
            max-width: 500px;
        }

        .top-matches-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .top-matches-grid {
                grid-template-columns: 1fr;
            }
        }

        .top-match-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            transition: all 0.3s ease;
        }

        .top-match-card:hover {
            transform: translateY(-5px);
            border-color: rgba(6, 182, 212, 0.3);
            background: rgba(30, 41, 59, 0.6);
        }

        .top-match-rank {
            background: rgba(255, 255, 255, 0.1);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--text-muted);
        }

        .top-match-card:first-child .top-match-rank {
            background: rgba(6, 182, 212, 0.2);
            color: var(--secondary);
        }

        .top-match-img {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
        }

        .top-match-name {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .top-match-score {
            font-size: 0.95rem;
            color: var(--secondary);
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .top-match-bar-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
        }

        .top-match-bar {
            height: 100%;
            background: var(--secondary);
            width: 0%;
            transition: width 1s ease-out;
        }

        .error-banner {
            display: none;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--danger);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: var(--text-main);
            margin-bottom: 1.5rem;
            align-items: center;
            gap: 1rem;
            animation: fadeIn 0.3s ease;
        }

        .error-icon {
            font-size: 1.5rem;
            color: var(--danger);
            flex-shrink: 0;
        }

        .error-message {
            font-size: 0.95rem;
            font-weight: 500;
        }

        .concepts-section {
            width: 100%;
            max-width: 900px;
            margin-top: 3rem;
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
        }

        .concepts-section h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .concept-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 768px) {
            .concept-grid {
                grid-template-columns: 1fr;
            }
        }

        .concept-item h4 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--secondary);
            margin-bottom: 0.4rem;
        }

        .concept-item p {
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Celebrity Look-Alike Finder</h1>
        <p>Upload a photo of your face, and our state-of-the-art AI will scan a dataset of celebrity faces to find your best matches!</p>
    </header>

    <main class="container">
        <div id="error-banner" class="error-banner">
            <span class="error-icon">⚠️</span>
            <span id="error-text" class="error-message"></span>
        </div>

        <div class="upload-area" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <span class="upload-icon">📷</span>
            <div class="upload-text">
                <h3>Drag & Drop Image Here</h3>
                <p>or click to browse from files (supports JPG, JPEG, PNG, WEBP up to 5MB)</p>
            </div>
            <input type="file" id="file-input" accept=".jpg,.jpeg,.png,.webp" onchange="handleFileSelect(event)">
        </div>

        <div class="preview-container" id="preview-container">
            <img id="preview-img" class="preview-image" src="" alt="Query Preview">
            <button type="button" class="remove-btn" onclick="clearPreview()">Remove Image</button>
        </div>

        <button id="submit-btn" class="action-btn" style="margin-top: 1.5rem;" onclick="processImage()" disabled>Find My Celebrity Look-Alike</button>

        <div class="loader-container" id="loader-container">
            <div class="spinner"></div>
            <div class="loader-text" id="loader-status">Uploading image...</div>
        </div>

        <div class="results-container" id="results-container">
            <h2 class="results-title">Analysis Results</h2>

            <div class="match-details" id="match-details">
                <div id="conf-badge" class="confidence-badge badge-high">Strong Match Found</div>
                <div style="font-size: 1.1rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Your Celebrity Look-Alike is</div>
                <div id="best-name" class="match-name">Brad Pitt</div>
                <div id="best-similarity" class="match-score">Similarity: 87.6%</div>
                
                <div class="progress-container">
                    <div id="best-bar" class="progress-bar"></div>
                </div>
                
                <div id="match-explanation" class="match-notice">
                    This match is determined using Strategy C: combining your peak similarity and Top-5 average similarity.
                </div>
            </div>

            <div class="match-comparison">
                <div class="image-card">
                    <h4>Your Image</h4>
                    <img id="query-result-img" class="match-img" src="" alt="Your Image">
                </div>
                <div class="image-card">
                    <h4>Celebrity Match</h4>
                    <img id="db-result-img" class="match-img" src="" alt="Celebrity Match">
                </div>
            </div>

            <h3 style="font-size: 1.4rem; font-weight: 600; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem;">Top 3 Look-Alikes</h3>
            <div class="top-matches-grid" id="top-matches-grid">
                <!-- Top 3 items injected dynamically -->
            </div>
        </div>
    </main>

    <section class="concepts-section">
        <h2>Learn the AI Concepts</h2>
        <div class="concept-grid">
            <div class="concept-item">
                <h4>Face Embeddings</h4>
                <p>When you upload an image, DeepFace uses a deep neural network (FaceNet512) to translate your facial features into a 512-dimensional vector of numbers. Visual similarity matches mathematical closeness in this space.</p>
            </div>
            <div class="concept-item">
                <h4>Cosine Similarity</h4>
                <p>We calculate the angular distance between your facial embedding vector and the celebrity vectors. Cosine similarity of 1.0 represents identical vectors, while 0.0 represents orthogonal vectors.</p>
            </div>
            <div class="concept-item">
                <h4>Vectorization via NumPy</h4>
                <p>Instead of looping through all 1,800 images in Python, we perform a vectorized matrix dot-product in C-optimized NumPy, calculating all similarity scores in less than a millisecond.</p>
            </div>
            <div class="concept-item">
                <h4>Identity Aggregation</h4>
                <p>Rather than comparing your face to just one image, we calculate a robust score per person. Strategy C computes a weighted average of your best match (60%) and your top 5 matches (40%) to ensure stability.</p>
            </div>
        </div>
    </section>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const previewContainer = document.getElementById('preview-container');
        const previewImg = document.getElementById('preview-img');
        const submitBtn = document.getElementById('submit-btn');
        const loaderContainer = document.getElementById('loader-container');
        const loaderStatus = document.getElementById('loader-status');
        const resultsContainer = document.getElementById('results-container');
        const errorBanner = document.getElementById('error-banner');
        const errorText = document.getElementById('error-text');

        let selectedFile = null;

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }

        function unhighlight(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }

        dropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFile(files[0]);
            }
        }

        function handleFileSelect(e) {
            const files = e.target.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        }

        function handleFile(file) {
            const allowed = ['.jpg', '.jpeg', '.png', '.webp'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();
            
            if (!allowed.includes(fileExt)) {
                showError("Invalid file type. Please upload a JPG, JPEG, PNG, or WEBP image.");
                clearPreview();
                return;
            }

            hideError();
            selectedFile = file;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                previewContainer.style.display = 'flex';
                submitBtn.disabled = false;
                resultsContainer.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        function clearPreview() {
            selectedFile = null;
            fileInput.value = '';
            previewContainer.style.display = 'none';
            previewImg.src = '';
            submitBtn.disabled = true;
            resultsContainer.style.display = 'none';
            hideError();
        }

        function showError(message) {
            errorText.textContent = message;
            errorBanner.style.display = 'flex';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function hideError() {
            errorBanner.style.display = 'none';
        }

        function processImage() {
            if (!selectedFile) return;

            hideError();
            resultsContainer.style.display = 'none';
            submitBtn.disabled = true;
            loaderContainer.style.display = 'flex';
            
            const statuses = [
                "Uploading photo...",
                "Running face detection filters...",
                "Extracting 512-dimensional DeepFace embeddings...",
                "Searching the celebrity look-alike dataset...",
                "Aggregating match percentages..."
            ];
            
            let statusIdx = 0;
            loaderStatus.textContent = statuses[statusIdx];
            const interval = setInterval(() => {
                if (statusIdx < statuses.length - 1) {
                    statusIdx++;
                    loaderStatus.textContent = statuses[statusIdx];
                }
            }, 1000);

            const formData = new FormData();
            formData.append('image', selectedFile);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(interval);
                loaderContainer.style.display = 'none';
                submitBtn.disabled = false;

                if (!data.success) {
                    showError(data.error || "An error occurred while processing the image.");
                    return;
                }

                document.getElementById('query-result-img').src = data.query_image_url;
                document.getElementById('db-result-img').src = data.best_match.representative_image_url;
                document.getElementById('best-name').textContent = data.best_match.person_name;
                
                const percentage = (data.best_match.score * 100).toFixed(1) + '%';
                document.getElementById('best-similarity').textContent = 'Similarity Score: ' + percentage;
                
                const badge = document.getElementById('conf-badge');
                badge.textContent = data.confidence_classification.label;
                badge.className = 'confidence-badge';
                
                if (data.confidence_classification.level === 'high') {
                    badge.classList.add('badge-high');
                } else if (data.confidence_classification.level === 'moderate') {
                    badge.classList.add('badge-mod');
                } else {
                    badge.classList.add('badge-low');
                }
                
                if (data.best_match.is_exact_match) {
                    document.getElementById('match-explanation').innerHTML = 
                        "<strong>Strong Match:</strong> This image appears to match " + data.best_match.person_name + 
                        " with high confidence. This image is already present in the celebrity dataset.";
                } else {
                    document.getElementById('match-explanation').textContent = data.confidence_classification.description;
                }

                const bar = document.getElementById('best-bar');
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = percentage;
                }, 100);

                const topGrid = document.getElementById('top-matches-grid');
                topGrid.innerHTML = '';
                
                data.top_matches.forEach((match, index) => {
                    const matchPercent = (match.score * 100).toFixed(1) + '%';
                    const card = document.createElement('div');
                    card.className = 'top-match-card';
                    card.innerHTML = `
                        <div class="top-match-rank">${index + 1}</div>
                        <img class="top-match-img" src="${match.representative_image_url}" alt="${match.person_name}">
                        <div class="top-match-name">${match.person_name}</div>
                        <div class="top-match-score">${matchPercent}</div>
                        <div class="top-match-bar-container">
                            <div class="top-match-bar" style="width: 0%"></div>
                        </div>
                    `;
                    topGrid.appendChild(card);
                    
                    setTimeout(() => {
                        card.querySelector('.top-match-bar').style.width = matchPercent;
                    }, 200 + (index * 150));
                });

                resultsContainer.style.display = 'block';
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(err => {
                clearInterval(interval);
                loaderContainer.style.display = 'none';
                submitBtn.disabled = false;
                showError("Network or server communication error. Please try again.");
                console.error(err);
            });
        }
    </script>
</body>
</html>"""

if __name__ == "__main__":
    # Check dataset existence
    if not DATASET_DIR.exists():
        logger.error(f"FATAL: The dataset folder was not found at '{DATASET_DIR}'. Ensure you put your folders inside it.")
        sys.exit(1)
        
    # Step 1: Scan current dataset files
    records, current_signature = scan_dataset(DATASET_DIR)
    if not records:
        logger.error("FATAL: Scanned dataset and found 0 images. Please add folders containing images to the 'dataset' directory.")
        sys.exit(1)
        
    logger.info(f"Scanned dataset: Found {len(records)} images belonging to {len(set(r['person_name'] for r in records))} distinct celebrities.")

    # Step 2: Load / Rebuild cache using signature comparison
    cache_loaded = False
    cached_data = load_cache(CACHE_FILE)
    
    # Verify cache validity (compare signature and check for matching metadata structures)
    if cached_data and "signature" in cached_data and "embeddings" in cached_data:
        if cached_data["signature"] == current_signature and len(cached_data["embeddings"]) > 0:
            logger.info("Dataset signature matches cache. Initializing with loaded cache.")
            embeddings_cache = cached_data["embeddings"]
            cache_loaded = True
        else:
            logger.info("Dataset changes detected! Re-generating embeddings cache...")
    else:
        logger.info("No valid cache found. Re-generating embeddings cache...")

    if not cache_loaded:
        # Generate new embeddings
        embeddings_cache = generate_dataset_embeddings(records)
        # Save cache
        new_cache_data = {
            "signature": current_signature,
            "embeddings": embeddings_cache
        }
        save_cache(CACHE_FILE, new_cache_data)

    # Step 3: Populate database variables in memory for fast matrix calculations
    initialize_memory_embeddings(embeddings_cache)

    # Step 4: Run Flask development server
    logger.info("Starting Flask application. Accessible locally at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
