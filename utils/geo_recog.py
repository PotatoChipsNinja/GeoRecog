import json
import torch
import threading
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

class OOVProcessor:
    def __init__(self, vocabs, emb_model_path):
        self.vocabs = vocabs
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
        self.model = AutoModel.from_pretrained(emb_model_path)
        self.model.eval()
        self.vocabs_emb = self.get_embeddings(vocabs)

    def get_embeddings(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def attach(self, text):
        targets = []
        for vocab in self.vocabs:
            if vocab in text or text in vocab:
                targets.append(vocab)
        if len(targets) == 1:
            return targets[0], 1.0
        else:
            emb = self.get_embeddings(text)
            similarities = emb @ self.vocabs_emb.T
            target = self.vocabs[similarities.argmax().item()]
            target_sim = similarities.max().item()
            return target, target_sim

class GeoRecog:
    def __init__(self, api_pool):
        self.api_pool = api_pool
        self.api_locks = [threading.Lock() for _ in api_pool]

        geo_dict_path = 'assets/geo_dict/all.txt'
        province_dict_path = 'assets/geo_dict/province.txt'
        self.name2code = {}
        with open(geo_dict_path, 'r') as f:
            all_vocabs = []
            for line in f.readlines():
                code, name = line.strip().split('=')
                self.name2code[name] = code
                all_vocabs.append(name)
        with open(province_dict_path, 'r') as f:
            province_vocabs = []
            for line in f.readlines():
                code, name = line.strip().split('=')
                self.name2code[name] = code
                province_vocabs.append(name)

        emb_model_path = 'assets/pretrained/bge-large-zh-v1.5'
        self.province_oov_processor = OOVProcessor(province_vocabs, emb_model_path)
        self.city_oov_processor = OOVProcessor(all_vocabs, emb_model_path)

    def get_api(self):
        while True:
            for i, api_url in enumerate(self.api_pool):
                if self.api_locks[i].acquire(timeout=0.02):
                    return api_url, self.api_locks[i]

    def get_api_response(self, messages):
        api_url, api_lock = self.get_api()
        client = OpenAI(base_url=api_url, api_key='None')
        response = client.chat.completions.create(
            model="Qwen2-7B-Instruct",
            messages=messages,
            temperature=0,
            timeout=5
        )
        ans = response.choices[0].message.content
        api_lock.release()
        return ans

    def llm_geo_recog(self, content):
        messages=[
            {"role": "system", "content": "请你从以下新闻内容中抽取出所有包含地理位置信息的命名实体（entities字段），并综合分析哪个实体最能反映新闻发生的属地（analysis字段），并给出新闻所处的省级行政区（province字段）和市级行政区（city字段）。直辖市的省级行政区和市级行政区相同。结果以JSON列表形式输出。"},
            {"role": "user", "content": "【#南昌市信访局通报一公职人员开车顶人#】19日，江西南昌市信访局发布情况通报，南昌市信访局关注到有媒体报道，反映该局一名公职人员吴某在上海南京路驾驶私家车时，将2名拦在其车辆前方的物业公司工作人员顶住前行，造成不良影响。据初步了解，此事发生在6月15日，现当事人吴某已向物业公司工作人员道歉并达成和解。针对吴某的不当行为，南昌市信访局会同有关部门正在深入调查。下一步，将依据调查结果严肃认真处理。（南昌信访）"},
            {"role": "assistant", "content": '{"entities": ["南昌市信访局", "上海南京路"], "analysis": "新闻中报道的通报由南昌市信访局发布，上海南京路只是通报事件的属地而非该新闻的属地，因此新闻属地为江西南昌。", "province": "江西", "city": "南昌"}'},
            {"role": "user", "content": "海洋一号D星。中新网北京6月11日电(郭超凯)记者从中国国家航天局获悉，6月11日2时31分，在牛家村，中国在太原卫星发射中心用长征二号丙运载火箭成功发射海洋一号D星。该星将与海洋一号C星组成中国首个海洋民用业务卫星星座。相比于美国，海洋一号D星是中国第四颗海洋水色系列卫星，是国家民用空间基础设施规划的首批海洋业务卫星之一。"},
            {"role": "assistant", "content": '{"entities": ["中国国家航天局", "牛家村", "太原卫星发射中心"], "analysis": "新闻中提及的发射地点为牛家村，发射中心为太原卫星发射中心，因此新闻属地为山西太原。", "province": "山西", "city": "太原"}'},
            {"role": "user", "content": "小李家住天通苑，是一名大学生。他于7月初到杭州西湖旅游，回到家后发现家里着火，立即拨打了119电话求救。消防员赶到后，发现火势已经蔓延到整栋楼，经过近两个小时的扑救，终于将火扑灭。经过调查，火灾原因是小李家里的电热水器短路引起的。"},
            {"role": "assistant", "content": '{"entities": ["天通苑", "杭州西湖"], "analysis": "新闻中提到的天通苑和杭州西湖分别是小李的居住地和旅游地，但新闻的主要内容是关于火灾事件，火灾发生地为小李的家中。因此，新闻的属地应为小李的居住地，即天通苑。", "province": "北京", "city": "北京"}'},
            {"role": "user", "content": content},
        ]
        try:
            res = self.get_api_response(messages)
            res = json.loads(res)
            return {
                'province': res['province'] if ('province' in res and len(res['province']) > 0) else None,
                'city': res['city'] if ('city' in res and len(res['city']) > 0) else None
            }
        except:
            return {'province': None, 'city': None}

    def query(self, content):
        llm_res = self.llm_geo_recog(content)
        if llm_res['province'] is not None:
            target, target_sim = self.province_oov_processor.attach(llm_res['province'])
            if target_sim > 0.8:
                llm_res['province'] = target
            else:
                llm_res['province'] = None
                llm_res['city'] = None
        if llm_res['city'] is not None:
            target, target_sim = self.city_oov_processor.attach(llm_res['city'])
            if target_sim > 0.8:
                llm_res['city'] = target
            else:
                llm_res['city'] = None
        llm_res['code'] = self.name2code.get(llm_res['city'], self.name2code.get(llm_res['province'], None))
        return llm_res
