sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install -r EasyEdit/requirements.txt
huggingface-cli download OpenMeditron/Meditron3-8B --local-dir /mloscratch/homes/blecoeur/hugging_cache/llama-3-8b-instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /mloscratch/homes/blecoeur/hugging_cache/real-llama-3-8b-instruct

