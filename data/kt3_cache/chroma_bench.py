import sys, json, time, resource
import numpy as np, psutil, chromadb
emb, timing, goldq, m, efc, efs, frac, out = sys.argv[1:]
E, Qt, Qg = np.load(emb), np.load(timing), np.load(goldq)
ids = np.random.default_rng(42).permutation(len(E))[:max(1, round(float(frac) * len(E)))]
proc = psutil.Process()
col = chromadb.EphemeralClient().create_collection('bench', embedding_function=None, configuration={'hnsw': {
    'space': 'cosine', 'max_neighbors': int(m), 'ef_construction': int(efc), 'ef_search': int(efs)}})
base = proc.memory_info().rss
t0 = time.time()
for i in range(0, len(ids), 1000):
    part = ids[i:i + 1000]
    col.add(ids=[str(j) for j in part], embeddings=E[part])
index_s = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
n_results = min(50, len(ids))
col.query(query_embeddings=Qt[:1], n_results=n_results)
times = []
for q in Qt:
    t0 = time.perf_counter()
    col.query(query_embeddings=[q], n_results=n_results, include=['distances'])
    times.append(time.perf_counter() - t0)
res = col.query(query_embeddings=Qg, n_results=n_results, include=['distances'])
json.dump({'n': len(ids), 'index_s': index_s, 'query_ms': 1000 * float(np.median(times)), 'rss_base_mb': base / 2**20,
           'rss_peak_mb': peak / 2**20, 'ids': [[int(x) for x in r] for r in res['ids']], 'dist': res['distances']}, open(out, 'w'))
