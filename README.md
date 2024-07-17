# 属地识别接口
## 环境要求
- Python >= 3.8, <= 3.11
- CUDA >= 12.0
- vLLM >= 0.5.1
- FastAPI >= 0.111.0

## 部署
- 拉取代码
  ``` shell
  git clone https://github.com/PotatoChipsNinja/GeoRecog.git
  cd GeoRecog
  ```
- 下载 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)、[bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) 模型
  ``` shell
  huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct --local-dir ./assets/pretrained/Qwen2-7B-Instruct
  huggingface-cli download --resume-download BAAI/bge-large-zh-v1.5 --local-dir ./assets/pretrained/bge-large-zh-v1.5
  ```
- 启动 uvicorn 服务
  ``` shell
  export GPU_NUM=4
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

## 接口说明
### 接口描述
接口URL：`http://127.0.0.1:8000/query`

请求方式：`GET`

### 请求参数
| 参数名称 | 必选 | 类型 | 说明 |
| :-: | :-: | :-: | :-: |
| content | 是 | String | 待检测内容 |

### 返回结果参数
| 参数 | 类型 | 说明 |
| :- | :-: | :-: |
| province | String | 省级行政区 |
| city | String | 地级市级行政区 |
| code | String | 行政区代码 |

### 调用示例
``` python
import json
import requests

api_url = 'http://127.0.0.1:8000/query'
data = { "content": "【#女子称养父去世后堂哥要求继承房产# 村支书回应】近日，河南周口一女子发视频称，堂哥“霸占”养父的房子土地。女子表示，养父2019年去世，因自己是女性，由堂哥在葬礼仪式上摔盆送养父下葬，于是堂哥要求继承养父的房子和土地。" }
res = requests.get(api_url, params=data)
res = json.loads(res.content)

print(res)  # {'province': '河南', 'city': '周口市', 'code': '411600'}
```

## 性能测试
| GPU | 平均时间 (s) | QPS |
|:-:|:-:|:-:|
| 1 * A800 80G | 0.7034 | 1.4216 |
| 2 * A800 80G | 0.3573 | 2.7990 |
| 4 * A800 80G | 0.1867 | 5.3550 |
| 8 * A800 80G | 0.0959 | 10.4266 |
