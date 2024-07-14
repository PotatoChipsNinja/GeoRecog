import os
import time
import requests
import subprocess

vLLM_processes = []

def init_vLLM():
    gpu_num = int(os.environ.get("GPU_NUM", 4))
    llm_path = "assets/pretrained/Qwen2-7B-Instruct"
    if not os.path.exists(os.path.join(llm_path, "config.json")):
        print(f"LLM not found, please check the path {llm_path}.")
        exit(-1)
    for i in range(gpu_num):
        port = 10800 + i
        cmd = f"CUDA_VISIBLE_DEVICES={i} python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model {llm_path} --port {port}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vLLM_processes.append(process)

    # Wait for the server to start
    time.sleep(10)
    remain_try = 20
    api_pool = []
    for i in range(gpu_num):
        port = 10800 + i
        while remain_try > 0:
            try:
                response = requests.get(f"http://localhost:{port}/v1/models")
                if response.status_code == 200:
                    api_pool.append(f"http://localhost:{port}/v1")
                    break
                else:
                    # throw exception
                    response.raise_for_status()
            except:
                remain_try -= 1
                if remain_try > 0:
                    time.sleep(5)
    if remain_try <= 0:
        print("vLLM server failed to start.")
        exit(-1)
    return api_pool

def stop_vLLM():
    for process in vLLM_processes:
        process.terminate()
