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
| province | String | 主属地省级行政区 |
| city | String | 主属地地级市级行政区 |
| code | String | 主属地行政区代码 |
| candidate | Array | 候选属地列表 |
| &rarr;province | String | 候选属地省级行政区 |
| &rarr;city | String | 候选属地地级市级行政区 |
| &rarr;code | String | 候选属地行政区代码 |

### 调用示例
``` python
import json
import requests

api_url = 'http://127.0.0.1:8000/query'
content = "在黔北深山，有位叫黄大发的老支书，今年82岁了。他用36年时间修渠，最终让全村人喝上了水。这位老支书很少出远门，两年前他第一次到贵州省城，他哪里也没去，只是到省委看国旗。1他第一次来北京，第一次看到天安门，不禁流下了泪水…#定格# "
data = { "content": content }
res = requests.get(api_url, params=data)
res = json.loads(res.content)

print(res)
```
返回结果如下：
``` json
{
    "province": "北京",
    "city": "北京市",
    "code": "110000",
    "candidate": [
        {
            "province": "贵州",
            "city": "贵阳市",
            "code": "520100"
        },
        {
            "province": "北京",
            "city": "北京市",
            "code": "110000"
        },
        {
            "province": "贵州",
            "city": "遵义市",
            "code": "520300"
        }
    ]
}
```

## 性能测试
| GPU | 平均时间 (s) | QPS |
|:-:|:-:|:-:|
| 1 * A800 80G | 0.7034 | 1.4216 |
| 2 * A800 80G | 0.3573 | 2.7990 |
| 4 * A800 80G | 0.1867 | 5.3550 |
| 8 * A800 80G | 0.0959 | 10.4266 |
