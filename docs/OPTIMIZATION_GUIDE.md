# Optimization Guide

**For:** CareerLens AI  
**Version:** v0.28.0  
**Last Updated:** March 15, 2026

---

## Current Performance Metrics

### ✅ What's Already Optimized

1. **Embedding Caching**
   - LRU cache for embeddings (500 max size)
   - Reduces redundant LLM calls
   - ~70% cache hit rate in testing

2. **Efficient Matching**
   - Cosine similarity for semantic matching
   - Normalized scoring (60/25/15 weights)
   - Case-insensitive skill matching

3. **Text Preprocessing**
   - Efficient tokenization
   - Stop word removal
   - Minimal string operations

4. **Memory Management**
   - No memory leaks detected
   - Garbage collection optimization
   - Efficient data structures

---

## Performance Bottlenecks

### Identified Bottlenecks

1. **LLM Calls (Ollama)**
   - **Impact:** 10-25s per call
   - **Frequency:** CV analysis, interview questions, learning pathways
   - **Solution:** Caching, async processing

2. **File Parsing (Large CVs)**
   - **Impact:** 3-8s for 500KB+ files
   - **Frequency:** Every CV upload
   - **Solution:** Chunking, streaming

3. **Concurrent Requests**
   - **Impact:** Degrades beyond 20 users
   - **Frequency:** Peak usage
   - **Solution:** Request queuing, load balancing

---

## Optimization Strategies

### Short-Term (Days 29-30)

#### 1. Response Caching

```python
# Cache common CV-JD pairs
from functools import lru_cache

@lru_cache(maxsize=100)
def compute_match_cached(cv_hash, jd_hash):
    return scoring_engine.compute_match_score(cv_data, jd_data)
```

#### 2. Async LLM Calls

```python
# For multiple LLM operations
import asyncio

async def generate_all_guidance(cv_data, jd_data):
    tasks = [
        analyze_cv_async(cv_data),
        generate_questions_async(cv_data),
        generate_pathway_async(cv_data, jd_data)
    ]
    return await asyncio.gather(*tasks)
```

#### 3. Request Queuing

```python
# Limit concurrent LLM calls
from queue import Queue
import threading

llm_queue = Queue(maxsize=5)

def process_llm_request(request):
    llm_queue.put(request)
    # Process when slot available
```

### Mid-Term (Deployment)

#### 1. Redis Caching

```python
import redis

# Cache scoring results
r = redis.Redis(host='localhost', port=6379)

def get_cached_score(cv_hash, jd_hash):
    key = f"match:{cv_hash}:{jd_hash}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)

    result = compute_match(cv_data, jd_data)
    r.setex(key, 3600, json.dumps(result))  # 1 hour TTL
    return result
```

#### 2. Database for Results

```python
# Store match results
from sqlalchemy import create_engine

# Avoid re-processing same CV-JD pairs
def check_existing_match(cv_id, jd_id):
    return db.query(Match).filter_by(
        cv_id=cv_id, jd_id=jd_id
    ).first()
```

#### 3. CDN for Static Assets

- Host Streamlit static files on CDN
- Reduce server load
- Faster page loads

### Long-Term (Scaling)

#### 1. Microservices Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Web App   │────▶│  API Gateway │────▶│  Parsers   │
└─────────────┘     └──────────────┘     └────────────┘
                           │
                           ├────▶ Scoring Service
                           │
                           ├────▶ LLM Service
                           │
                           └────▶ Cache Service
```

#### 2. Load Balancing

```nginx
upstream careerlens_backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    location / {
        proxy_pass http://careerlens_backend;
    }
}
```

#### 3. Horizontal Scaling

- Multiple Streamlit instances
- Shared Redis cache
- Load-balanced requests

---

## Caching Strategy

### What to Cache

| Data          | TTL       | Size Limit | Priority |
| ------------- | --------- | ---------- | -------- |
| Embeddings    | Permanent | 500 items  | High     |
| Match Scores  | 1 hour    | 100 pairs  | High     |
| LLM Responses | 24 hours  | 50 items   | Medium   |
| Parsed CVs    | 1 hour    | 20 items   | Low      |

### Cache Implementation

```python
from cachetools import TTLCache
import hashlib

# Match score cache
match_cache = TTLCache(maxsize=100, ttl=3600)

def get_match_score(cv_data, jd_data):
    # Create cache key
    cv_hash = hashlib.md5(cv_data['text'].encode()).hexdigest()
    jd_hash = hashlib.md5(jd_data['text'].encode()).hexdigest()
    cache_key = f"{cv_hash}:{jd_hash}"

    # Check cache
    if cache_key in match_cache:
        return match_cache[cache_key]

    # Compute and cache
    result = scoring_engine.compute_match_score(cv_data, jd_data)
    match_cache[cache_key] = result

    return result
```

---

## Database Optimization

### If Using Database

```sql
-- Index for fast lookups
CREATE INDEX idx_cv_jd ON matches(cv_id, jd_id);
CREATE INDEX idx_created_at ON matches(created_at);

-- Partition by date
CREATE TABLE matches_2026_03 PARTITION OF matches
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
```

### Query Optimization

```python
# Use select_related for joins
matches = Match.objects.select_related('cv', 'jd').filter(
    created_at__gte=datetime.now() - timedelta(days=7)
)

# Limit fields
matches = Match.objects.only('score', 'created_at').all()
```

---

## Frontend Optimization

### Streamlit Best Practices

```python
# Use @st.cache_data for data
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

# Use @st.cache_resource for models
@st.cache_resource
def load_model():
    return ScoringEngine()

# Lazy loading
with st.spinner('Loading...'):
    result = compute_expensive_operation()
```

---

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Response Time**
   - Alert if avg > 5s (non-LLM)
   - Alert if avg > 20s (LLM)

2. **Error Rate**
   - Alert if > 5%

3. **Memory Usage**
   - Alert if > 80% of available RAM

4. **Throughput**
   - Alert if < 1 request/second

### Implementation

```python
import logging
import time

logger = logging.getLogger(__name__)

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            # Log performance
            logger.info(f"{func.__name__}: {elapsed:.2f}s")

            # Alert if slow
            if elapsed > 5.0:
                logger.warning(f"{func.__name__} slow: {elapsed:.2f}s")

            return result
        except Exception as e:
            logger.error(f"{func.__name__} error: {e}")
            raise

    return wrapper
```

---

## Testing Performance Changes

### Before/After Comparison

```bash
# Baseline benchmark
python scripts/benchmark.py > baseline.txt

# Make optimization changes

# New benchmark
python scripts/benchmark.py > optimized.txt

# Compare
diff baseline.txt optimized.txt
```

---

## Recommended Tools

- **Profiling:** `cProfile`, `memory_profiler`
- **Monitoring:** Prometheus, Grafana
- **Caching:** Redis
- **Load Testing:** Locust, Apache JMeter
- **APM:** New Relic, Datadog

---

**Next Review:** After deployment  
**Target Metrics:** < 3s response, 20+ concurrent users
