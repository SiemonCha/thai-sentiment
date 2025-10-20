from huggingface_hub import HfApi, create_repo, list_repo_files
import os

USERNAME = "SiemonCha"
REPO_NAME = "thai-sentiment-phayabert"
repo_id = f"{USERNAME}/{REPO_NAME}"

api = HfApi()

# Check auth
try:
    user = api.whoami()
    print(f">>> Logged in as: {user['name']}")
except:
    print(">>> Run: huggingface-cli login")
    exit(1)

# Create repo
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f">>> Repo: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f">>> {e}")

# Check files
model_path = "./model"
if not os.path.exists(f"{model_path}/config.json"):
    print("XXX No model found. Run: python train.py")
    exit(1)

# Upload
print("\n--- Uploading...")
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_id,
    repo_type="model"
)

print(f"\n>>> Done! https://huggingface.co/{repo_id}")

# Verify
files = list_repo_files(repo_id, repo_type="model")
print(f"\nUploaded {len(files)} files:")
for f in files[:10]:
    print(f"  ✓ {f}")