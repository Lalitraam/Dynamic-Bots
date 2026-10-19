import sys
import time
import subprocess
from pathlib import Path
from app.db.migrate import get_db
from app.db.jobs import claim_next_job, update_job, update_player
from app.services.downloader import download_player_games
from app.services.dataset_builder_v3 import build_dataset_v3
from app.ml.tensor_cache import build_cache

def process_job(conn, job):
    job_id = job['id']
    username = job['username']
    print(f"\n[Worker] ==============================================", flush=True)
    print(f"[Worker] Starting pipeline for '{username}' (Job ID: {job_id})", flush=True)
    print(f"[Worker] ==============================================", flush=True)
    
    update_player(conn, username, 'building')
    
    try:
        # Step 1: Download
        update_job(conn, job_id, step="Downloading games from Lichess", progress=0.1)
        print(f"\n[Worker] Step 1/4: Downloading games for {username}...", flush=True)
        res = download_player_games(username)
        if isinstance(res, dict):
            print(f"  -> Total games found: {res.get('total_games', 0)}", flush=True)
            print(f"  -> Games sampled: {res.get('sampled_games', 0)}", flush=True)
            drops = res.get('drop_stats', {})
            if drops:
                print(f"  -> Dropped games: {drops}", flush=True)
        
        # Step 2: Build Dataset
        update_job(conn, job_id, step="Building dataset", progress=0.3)
        print(f"\n[Worker] Step 2/4: Building v3 dataset...", flush=True)
        res = build_dataset_v3(username)
        if isinstance(res, dict):
            print(f"  -> Total positions: {res.get('total_samples', 0)}", flush=True)
            splits = res.get('split_counts', {})
            print(f"  -> Train: {splits.get('train', 0)} | Val: {splits.get('val', 0)} | Test: {splits.get('test', 0)}", flush=True)
        
        # Step 3: Build Cache
        update_job(conn, job_id, step="Building tensor cache", progress=0.5)
        print(f"\n[Worker] Step 3/4: Building tensor cache...", flush=True)
        cache_dir = build_cache(username)
        print(f"  -> Cache built at: {cache_dir}", flush=True)
        
        # Step 4: Fine-tune
        update_job(conn, job_id, step="Fine-tuning model (this takes a few minutes)", progress=0.7)
        print(f"\n[Worker] Step 4/4: Fine-tuning model on CPU...", flush=True)
        print(f"  -> Launching finetune_runner.py --max-minutes 10", flush=True)
        
        # Use sys.executable to ensure we use the same python
        cmd = [sys.executable, "scripts/finetune_runner.py", username, "--max-minutes", "10"]
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            raise RuntimeError(f"finetune_runner.py failed with exit code {result.returncode}")
        
        # Done
        update_job(conn, job_id, status='completed', step="Ready", progress=1.0)
        update_player(conn, username, 'ready')
        print(f"\n[Worker] Successfully completed '{username}'!", flush=True)
        print(f"[Worker] Bot is now READY. You can play it in the browser.", flush=True)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[Worker] ERROR processing '{username}': {error_msg}", flush=True)
        import traceback
        traceback.print_exc()
        update_job(conn, job_id, status='failed', step="Failed", error=error_msg)
        update_player(conn, username, 'failed', error=error_msg)

def main():
    print("Worker started. Polling for jobs...", flush=True)
    while True:
        try:
            conn = get_db()
            job = claim_next_job(conn)
            if job:
                process_job(conn, job)
            else:
                time.sleep(5)
            conn.close()
        except KeyboardInterrupt:
            print("Worker shutting down...", flush=True)
            break
        except Exception as e:
            print(f"Worker loop error: {e}", flush=True)
            time.sleep(5)

if __name__ == "__main__":
    main()
