# SigmaK: Architecture & Technical Reference

> For orientation and quick-start guides, see the [README](../README.md).

---

## System Architecture

The application follows a modular "Extraction-to-Storage" flow with **asynchronous task processing**:

1. **Ingestion Engine (`ingest.py`)**: Parses raw SEC HTM files using BeautifulSoup (lxml). Features robust encoding fallbacks (UTF-8/CP1252) and regex-based slicing to isolate "Item 1A."
2. **Processing Layer (`processing.py`)**: Implements recursive character text splitting. Chunks are normalized to ~800 characters with an 80-character overlap to preserve semantic context.
3. **Embedding Engine (`embeddings.py`)**: Converts text chunks into 384-dimensional semantic vectors using the `all-MiniLM-L6-v2` sentence transformer model.
4. **The Vault (`init_vector_db.py`)**: A persistent **ChromaDB** instance utilizing **HNSW** indexing with a **Cosine Similarity** metric. Storage path: `./database/`
5. **Reranking Layer (`reranking.py`)**: Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) that reranks search results for improved relevance. Processes query-document pairs jointly for superior accuracy.
6. **Risk Classification (`risk_taxonomy.py`, `prompt_manager.py`)**: Proprietary 10-category risk taxonomy with version-controlled LLM prompts for semantic classification.
7. **LLM Integration (`llm_classifier.py`, `llm_storage.py`, `risk_classifier.py`)**: Gemini 2.5 Flash Lite integration with threshold-based routing, intelligent caching, and full provenance tracking. Automatically falls back to LLM when vector search confidence is below 0.64, uses LLM confirmation for scores between 0.64–0.80, and directly uses vector results for scores ≥ 0.80.
8. **Drift Detection System (`drift_detection.py`, `drift_scheduler.py`)**: SQLite + ChromaDB dual-storage architecture with periodic review jobs to detect classification drift. Samples low-confidence and old classifications, re-runs LLM classification, and measures agreement rate. Triggers manual review alerts when drift exceeds thresholds (< 75% agreement).
9. **Risk Scoring (`scoring.py`)**: Quantifies **Severity** (0.0–1.0) and **Novelty** (0.0–1.0) for risk disclosures using keyword analysis and semantic comparison with historical filings. Every score includes full source citation and human-readable explanation.
10. **Orchestration Layer (`indexing_pipeline.py`)**: End-to-end pipeline coordinator that integrates extraction, chunking, embedding, storage, and hybrid search with optional reranking.
11. **Async Task Queue (`tasks.py`)**: Celery + Redis background task processor for non-blocking API operations. All slow operations (ingestion, scoring) run in worker processes with progress tracking.
12. **REST API (`api.py`)**: FastAPI server with async endpoints, authentication, rate limiting, and real-time task status polling.

---

## Storage & Retrieval

**Vector Database**: ChromaDB with SQLite backend
- **Storage Path**: `./database/` (configurable)
- **Collection**: `sec_risk_factors`
- **Distance Metric**: Cosine similarity (optimized for semantic search)
- **Index Type**: HNSW (Hierarchical Navigable Small World)

**Metadata Schema** (enforced on every chunk):
```python
{
    "ticker": str,        # Stock symbol (e.g., "AAPL")
    "filing_year": int,   # Filing year (e.g., 2025)
    "item_type": str      # SEC section (e.g., "Item 1A")
}
```

**Retrieval Logic**:
- **Semantic Search**: Natural language queries (e.g., "Geopolitical Instability") retrieve relevant chunks without exact keyword matches.
- **Hybrid Filtering**: Combine semantic search with metadata filters (e.g., "Find AAPL risks from 2025 related to supply chain").
- **Two-Stage Reranking** (Optional): Retrieve broader candidates via bi-encoder, then rerank with cross-encoder for maximum precision.
- **Deterministic IDs**: Document IDs follow `{ticker}_{year}_{chunk_index}` pattern for safe upserts.

**Performance Trade-offs**:
- **Vector-only**: ~150ms, good for exploratory queries
- **Reranked**: ~170ms, optimal for high-precision results
- Reranking adds ~20ms overhead but significantly improves relevance

---

## Configuration

SigmaK uses a YAML-based configuration system with environment variable overrides.

**Config File**: `config.yaml` at repository root

**Key Settings**:
```yaml
database:
  sqlite_path: "database/risk_classifications.db"

chroma:
  persist_directory: "database"
  embedding_model: "all-MiniLM-L6-v2"
  llm_cache_similarity_threshold: 0.8  # Reuse cached LLM if similarity >= 0.8

llm:
  model: "gemini-2.0-flash-exp"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
```

**Environment Overrides**:
- `SIGMAK_SQLITE_PATH` → database.sqlite_path
- `SIGMAK_LLM_MODEL` → llm.model
- `SIGMAK_EMBEDDING_MODEL` → chroma.embedding_model
- `LOG_LEVEL` → logging.level
- `CHROMA_PERSIST_PATH` → chroma.persist_directory
- `REDIS_URL` → Redis connection (default: `redis://localhost:6379/0`)
- `ENVIRONMENT` → deployment environment (default: `development`)

**Programmatic Access**:
```python
from sigmak.config import get_settings

settings = get_settings()
print(settings.chroma.llm_cache_similarity_threshold)  # 0.8
print(settings.database.sqlite_path)  # Path object
```

---

## Backfilling LLM Classifications

After analyzing filings, you can backfill existing LLM classifications into the vector database for future reuse:

```bash
# Preview what would be inserted (dry run)
uv run sigmak backfill --dry-run

# Write classifications to database
uv run sigmak backfill --write

# Custom paths
uv run sigmak backfill --write \
    --output-dir ./custom_output \
    --db-path ./database/custom.db
```

**What it does**:
- Reads all `results_*.json` files from `output/` directory
- Extracts LLM classification results (category, confidence, evidence, rationale, prompt_version)
- Generates embeddings for each risk text
- Inserts into SQLite audit table + ChromaDB collection
- Skips duplicates automatically (based on text hash)
- Reports statistics (files processed, entries inserted/skipped/errors)

**Why backfill**:
- Future classifications can reuse cached results via similarity search
- Reduces LLM API calls and costs when analyzing new filings
- Preserves full provenance (prompt_version, model_version, timestamp)
- Enables drift detection by comparing old vs new classifications

---

## Downloading 10-K Filings

The 10-K downloader retrieves filings directly from SEC EDGAR and tracks them in SQLite.

**Features**:
- **Automatic Ticker → CIK Resolution**: Resolves ticker symbols to SEC Central Index Keys using SEC's company tickers JSON
- **Intelligent Retry Logic**: Exponential backoff on transient errors (429 rate limiting, 503 service unavailable)
- **Dual SQLite Tracking**:
  - `filings_index`: Stores filing metadata (CIK, accession number, filing date, SEC URL)
  - `downloads`: Tracks downloaded files with SHA-256 checksums for integrity verification
- **Idempotent Downloads**: Skips re-downloading existing files unless `--force-refresh` is used
- **SEC Compliance**: Proper User-Agent header, respects rate limits, follows SEC EDGAR best practices

**Database Schema**:
```sql
-- Filing metadata (one row per unique SEC filing)
CREATE TABLE filings_index (
    id TEXT PRIMARY KEY,              -- UUID
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    accession TEXT NOT NULL,
    filing_type TEXT NOT NULL,
    filing_date TEXT,
    sec_url TEXT,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(cik, accession)
);

-- Downloaded files (one row per downloaded file)
CREATE TABLE downloads (
    id TEXT PRIMARY KEY,              -- UUID
    filing_index_id TEXT NOT NULL,
    local_path TEXT NOT NULL,
    download_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    http_status INTEGER,
    bytes INTEGER,
    checksum TEXT,                    -- SHA-256 for integrity
    FOREIGN KEY (filing_index_id) REFERENCES filings_index(id),
    UNIQUE(filing_index_id, local_path)
);
```

**Default Paths**:
- **Database**: `./database/sec_filings.db`
- **Downloads**: `./data/filings/{ticker}/{year}/`

**Programmatic Usage**:
```python
from sigmak.downloads import TenKDownloader

downloader = TenKDownloader(
    db_path="./database/sec_filings.db",
    download_dir="./data/filings"
)

records = downloader.download_10k(ticker="MSFT", years=3)
for record in records:
    print(f"{record.ticker} {record.year} → {record.local_path}")
```

---

## Programmatic Usage

### Indexing a Filing

```python
from sigmak.indexing_pipeline import IndexingPipeline

pipeline = IndexingPipeline(persist_path="./chroma_db")

stats = pipeline.index_filing(
    html_path="data/sample_10k.html",
    ticker="AAPL",
    filing_year=2025,
    item_type="Item 1A"
)

print(f"Indexed {stats['chunks_indexed']} chunks in {stats['embedding_latency_ms']:.2f}ms")
```

### Semantic Search

```python
# Basic vector search
results = pipeline.semantic_search(
    query="Geopolitical risks and international conflicts",
    n_results=5
)

# Hybrid search with metadata filtering
results = pipeline.semantic_search(
    query="Supply chain disruptions",
    n_results=5,
    where={"ticker": "AAPL", "filing_year": 2025}
)

for result in results:
    print(f"[{result['metadata']['ticker']}] {result['text'][:100]}...")
    print(f"Distance: {result['distance']:.4f}\n")
```

### Reranked Search (Higher Precision)

```python
results = pipeline.semantic_search(
    query="regulatory compliance risks in technology sector",
    n_results=3,
    rerank=True
)

for result in results:
    print(f"Vector Distance: {result['distance']:.4f}")
    print(f"Rerank Score: {result['rerank_score']:.4f}\n")
```

### Risk Scoring

```python
from sigmak.scoring import RiskScorer

scorer = RiskScorer()

chunk = {
    "text": "Catastrophic supply chain disruptions due to geopolitical conflicts",
    "metadata": {"ticker": "AAPL", "filing_year": 2025, "item_type": "Item 1A"}
}

severity = scorer.calculate_severity(chunk)
print(f"Severity: {severity.value:.2f}")  # 0.0–1.0, higher = more severe

historical_chunks = [
    {"text": "Standard competition risks", "metadata": {"ticker": "AAPL", "filing_year": 2024}}
]

novelty = scorer.calculate_novelty(chunk, historical_chunks)
print(f"Novelty: {novelty.value:.2f}")    # 0.0–1.0, higher = more novel
```

### Integration Pipeline (End-to-End)

```python
from sigmak.integration import IntegrationPipeline

pipeline = IntegrationPipeline(persist_path="./chroma_db")

result = pipeline.analyze_filing(
    html_path="data/sample_10k.html",
    ticker="AAPL",
    filing_year=2025,
    retrieve_top_k=10
)

for risk in result.risks:
    print(f"Severity: {risk['severity']['value']:.2f} | Novelty: {risk['novelty']['value']:.2f}")
    print(f"Text: {risk['text'][:100]}...")

# Export to JSON
with open("risk_analysis.json", "w") as f:
    f.write(result.to_json())
```

**Pipeline Flow**:
1. **Validation** — Check file exists, ticker format, year range
2. **Indexing** — Extract Item 1A → chunk → embed → store
3. **Retrieval** — Semantic search for top-k risk factors
4. **Scoring** — Compute severity and novelty for each
5. **Output** — Structured JSON with complete provenance

---

## REST API

### Starting the System

```bash
# 1. Start Redis (message broker)
redis-server

# 2. Start Celery worker
celery -A sigmak.tasks worker --loglevel=info

# 3. Start API server
uv run uvicorn sigmak.api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### `POST /analyze` — Submit analysis task (async)

**Authentication**: Requires `X-API-Key` header

**Request Body**:
```json
{
  "ticker": "AAPL",
  "filing_year": 2025,
  "html_path": "/path/to/filing.html",
  "retrieve_top_k": 10
}
```

**Response (HTTP 202 Accepted)**:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status_url": "/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Analysis task submitted successfully"
}
```

#### `GET /tasks/{task_id}` — Poll task status

**Task States**: `PENDING` → `PROGRESS` → `SUCCESS` | `FAILURE`

**Response (SUCCESS)**:
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "SUCCESS",
  "result": {
    "ticker": "AAPL",
    "filing_year": 2025,
    "risks": [
      {
        "text": "Supply chain disruptions could materially impact...",
        "severity": { "value": 0.75, "explanation": "High severity due to keywords: severe, disruption" },
        "novelty":  { "value": 0.82, "explanation": "High novelty — semantically distinct from 2024 filing" }
      }
    ],
    "metadata": { "total_latency_ms": 2534.5, "chunks_indexed": 5 }
  }
}
```

#### `POST /index` — Bulk ingestion (no scoring)

Same response format as `POST /analyze`. Use for pre-indexing filings before scoring.

#### `GET /health` — No authentication required

```json
{ "status": "healthy", "version": "1.0.0" }
```

### Authentication & Rate Limiting

```python
from sigmak.auth import APIKeyManager

manager = APIKeyManager()
api_key = manager.create_api_key(user="client_name", rate_limit="10/minute")
# Keys stored in api_keys.json
```

- Default: 10 requests/minute per API key
- Exceeding limit returns HTTP 429

### Validation Rules (Pydantic)

| Field | Rule |
|---|---|
| `ticker` | 1–10 uppercase alphanumeric characters |
| `filing_year` | 1994–2050 |
| `html_content` / `html_path` | Exactly one must be provided |
| `retrieve_top_k` | 1–100 (default: 10) |

Invalid requests return **422 Unprocessable Entity** with detailed error messages.

---

## Drift Detection System

### Architecture: Dual Storage

- **SQLite**: Stores full classification provenance (text, category, confidence, rationale, model version, source, timestamp)
- **ChromaDB**: Stores embeddings for semantic search, cross-referenced to SQLite via `chroma_id`
- **Deduplication**: Text hashing prevents duplicate storage
- **Archive Versioning**: Old embeddings are archived when embedding models change

**Classification Sources**: `LLM` (Gemini), `VECTOR` (ChromaDB similarity), `MANUAL` (human label)

### Drift Review Process

1. Sample low-confidence classifications (confidence < 0.75)
2. Sample old records (> 90 days)
3. Re-classify with LLM (current model)
4. Measure agreement rate
5. Log metrics and trigger alerts:
   - **WARNING** (< 85% agreement): Log warning
   - **CRITICAL** (< 75% agreement): Require manual review

### Running Drift Detection

**In-process (development)**:
```python
from sigmak.drift_scheduler import DriftScheduler

scheduler = DriftScheduler(
    db_path="./database/risk_classifications.db",
    chroma_path="./database"
)
scheduler.start()
scheduler.schedule_drift_review_cron(hour=2, minute=0, sample_size=20)
```

**Cron job (production)**:
```bash
0 2 * * * cd /app && python -m sigmak.drift_scheduler --run-once --sample-size 50
```

**Systemd timer**:
```ini
# /etc/systemd/system/sigmak-drift-review.timer
[Timer]
OnCalendar=02:00
Persistent=true
```

### Embedding Model Migration

```python
from sigmak.drift_detection import DriftDetectionSystem
from sigmak.embeddings import EmbeddingEngine

system = DriftDetectionSystem()
new_engine = EmbeddingEngine(model_name="all-MiniLM-L12-v2")

for record_id in range(1, 1001):
    record = system.get_record_by_id(record_id)
    if record:
        new_embedding = new_engine.encode([record['text']])[0].tolist()
        system.archive_and_update_embedding(
            record_id=record_id,
            new_embedding=new_embedding,
            new_model_version="all-MiniLM-L12-v2"
        )
```

---

## Deployment

### Docker Compose

```bash
docker-compose build
docker-compose up -d

# Check services
docker-compose ps
docker-compose logs -f api
```

**Services**: `api` (port 8000), `worker` (Celery), `redis` (broker + result backend)

**Environment variables**:
```bash
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
ENVIRONMENT=production
CHROMA_PERSIST_PATH=/app/chroma_db
```

**Health Checks**:
- `GET /health` — basic
- `GET /ready` — readiness (checks Redis, ChromaDB)
- `GET /live` — liveness (detects deadlocks)

### Cloud Deployment (Digital Ocean)

```bash
# 1. Create Droplet (Ubuntu 22.04 LTS, 4GB RAM minimum)
doctl compute droplet create sigmak \
  --image ubuntu-22-04-x64 \
  --size s-2vcpu-4gb \
  --region nyc3 \
  --ssh-keys your-ssh-key-id

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
apt-get install docker-compose-plugin -y

# 3. Clone and deploy
git clone https://github.com/your-username/sigmak.git && cd sigmak
docker-compose up -d

# 4. Firewall
ufw allow 22/tcp && ufw allow 8000/tcp && ufw enable
```

**HTTPS with Nginx** (recommended for production):
```bash
apt-get install nginx certbot python3-certbot-nginx -y
# Configure reverse proxy, then:
certbot --nginx -d your-domain.com
```

### Performance Tuning

```bash
# Celery: adjust concurrency
celery -A sigmak.tasks worker --concurrency=4

# API: Gunicorn multi-worker
gunicorn sigmak.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Redis memory (docker-compose.yml)**:
```yaml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Maintenance

```bash
# Backup ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./database/

# Update application
git pull origin main
docker-compose down && docker-compose build && docker-compose up -d

# Monitor resources
docker stats && docker system df
```

---

## Peer Discovery: yfinance Adapter (opt-in)

> **Demo / PoC only.** The canonical peer discovery path (SEC EDGAR + SIC matching) remains authoritative for production. This adapter is a fast, automatable alternative for demos.
> **Legal note**: `yfinance` wraps unofficial Yahoo Finance endpoints. Review Yahoo's Terms of Service before sustained production use.

### Enable

```bash
export SIGMAK_PEER_YFINANCE_ENABLED=true
```

### Usage

```python
from sigmak.peer_discovery import PeerDiscoveryService

svc = PeerDiscoveryService()
peers = svc.get_peers_via_yfinance("NVDA", n=10)
for p in peers:
    print(p.ticker, p.company_name, p.market_cap)
```

### Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `SIGMAK_PEER_YFINANCE_ENABLED` | `false` | Must be `true` to enable |
| `SIGMAK_PEER_YFINANCE_N_PEERS` | `10` | Max peers to return |
| `SIGMAK_PEER_YFINANCE_MIN_PEERS` | `5` | Min peers before threshold relaxation |
| `SIGMAK_PEER_YFINANCE_TTL_SECONDS` | `86400` | Cache TTL (24 h) |
| `SIGMAK_PEER_YFINANCE_MIN_FRACTION` | `0.10` | Min market-cap as fraction of target |
| `SIGMAK_PEER_YFINANCE_MIN_ABS_CAP` | `50000000` | Absolute market-cap floor ($50 M) |
| `SIGMAK_PEER_YFINANCE_RATE_LIMIT_RPS` | `1` | Soft rate limit (req/s) |
| `SIGMAK_PEER_YFINANCE_MAX_RETRIES` | `3` | Retries on transient errors |
| `SIGMAK_PEER_YFINANCE_BACKOFF_BASE` | `0.5` | Exponential backoff base (seconds) |

Cached payloads are written to `cache_dir/yfinance/` and are covered by `.gitignore`.

---

## Peer Discovery DB

A lightweight `peers` SQLite table stores peer metadata: ticker, CIK, SIC, industry, market_cap, company name, SIC description, state of incorporation, and recent filing dates. Market-cap upserts preserve existing values when an incoming update carries no market-cap data (prevents accidental NULL overwrites). `PeerDiscoveryService` prefers DB-first discovery to avoid thousands of live SEC calls.
