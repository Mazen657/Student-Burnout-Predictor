# ============================================================
#  DEPLOYMENT CELL — run this last in Google Colab
#  Requires: model_artifacts/ folder already saved
# ============================================================

# ── Step 1: Install dependencies ────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "streamlit", "pyngrok"], check=True)

# ── Step 2: Copy app.py from repo (or it's already there) ───
import os
print("✅ app.py ready" if os.path.exists("app.py") else "⚠️  app.py not found — make sure it's in the same directory")

# ── Step 3: Launch Streamlit ─────────────────────────────────
import threading, time

def _run_streamlit():
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])

t = threading.Thread(target=_run_streamlit, daemon=True)
t.start()
print("⏳ Waiting for Streamlit to start...")
time.sleep(12)

# ── Step 4: Open ngrok tunnel ────────────────────────────────
from pyngrok import ngrok
from google.colab import userdata

token = userdata.get('NGROK_AUTH_TOKEN')
if token:
    ngrok.set_auth_token(token)
    print("✅ ngrok token set")
else:
    raise ValueError("⚠️  Add NGROK_AUTH_TOKEN to Colab Secrets (key icon in sidebar)")

ngrok.kill()
public_url = ngrok.connect(8501)

print("\n" + "=" * 55)
print("🎉  App is live!")
print(f"🌐  {public_url}")
print("=" * 55)
print("⚠️  Keep this cell running — closing Colab stops the app")

# ── Keep alive ────────────────────────────────────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    ngrok.kill()
