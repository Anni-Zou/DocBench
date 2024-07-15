from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from secret_key import HF_KEY

parser = ArgumentParser()
parser.add_argument("--local_dir", default="YOUR OWN LOCAL DIR")
parser.add_argument("--repo_id", default="meta-llama/Meta-Llama-3-70B-Instruct")
args = parser.parse_args()


snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir+args.repo_id, local_dir_use_symlinks=False, token=HF_KEY)