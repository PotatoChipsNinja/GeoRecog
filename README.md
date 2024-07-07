# 属地识别接口
## 环境要求
- Python >= 3.8, <= 3.11
- CUDA >= 12.0
- vLLM >= 0.5.1
- FastAPI >= 0.111.0
- uvicorn >= 0.30.1

## 部署
- 下载 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 到本地
  ``` shell
  huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct --local-dir ./Qwen2-7B-Instruct
  ```
- 启动 uvicorn 服务
  ``` shell
  export GPU_NUM=4
  export LLM_PATH=./Qwen2-7B-Instruct
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

## 调用
``` python
import json
import requests

api_url = 'http://127.0.0.1:8000/query'
data = { "content": "我爱北京天安门" }
res = requests.get(api_url, params=data)
res = json.loads(res.content)

print(res)  # {'province': '北京', 'city': '北京'}
```

## 性能测试
| GPU | 平均时间 (s) | QPS |
|:-:|:-:|:-:|
| 1 * A800 80G | 0.7034 | 1.4216 |
| 2 * A800 80G | 0.3573 | 2.7990 |
| 4 * A800 80G | 0.1867 | 5.3550 |
| 8 * A800 80G | 0.0959 | 10.4266 |