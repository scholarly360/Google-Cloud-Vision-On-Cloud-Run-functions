# curl -LsSf https://astral.sh/uv/install.sh | sh
# uv init && uv add 
python -m pip install --upgrade pip
sudo echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
pip install --no-cache-dir -r requirements.txt