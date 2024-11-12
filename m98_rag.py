import os
import json
import re
import gzip
import urllib.request
import urllib.error


def post_json(url: str, data: dict, headers: dict, decoding='utf-8'):
    json_data = json.dumps(data)
    bytes_data = json_data.encode('utf-8')
    headers.update({'Content-Type': 'application/json'})
    request = urllib.request.Request(url, data=bytes_data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request) as response:
            response_content = response.read()
            return response_content.decode(decoding)
    except urllib.error.HTTPError as e:
        # 输出HTTP状态码和原因
        print(f"HTTP error occurred: {e.code} - {e.reason}")
        # 输出响应的错误内容（如果有的话）
        if e.fp:
            error_message = e.fp.read().decode('utf-8')
            print("Error message:", error_message)
            return error_message
    except urllib.error.URLError as e:
        # 输出与URL相关的错误
        print(f"URL error occurred: {e.reason}")
        return e.reason
    except Exception as e:
        # 输出其他所有错误
        raise e


QDRANT_EMBD_URL = os.getenv('QDRANT_EMBD_URL', 'http://localhost:8080/v1/embeddings')
QDRANT_EMBD_KEY = 'Bearer ' + os.getenv('QDRANT_EMBD_KEY', 'no-key')
QDRANT_EMBD_MODEL = os.getenv('QDRANT_EMBD_MODEL', 'text-embedding-3-small')


def embd(input_: list):
    response_content: str = post_json(url=QDRANT_EMBD_URL,
                                      data={
                                          'model': QDRANT_EMBD_MODEL,
                                          "encoding_format": "float",
                                          'input': input_
                                      },
                                      headers={'Authorization': QDRANT_EMBD_KEY})
    response: dict = json.loads(response_content)
    if type(response) is not dict or not response:
        print(response_content)
        return
    data: list = response.get('data', [])
    if not data:
        print(response_content)
        return
    if len(data) != len(input_):
        print(response_content)
        return
    return [(input_[i], data[i]['embedding']) for i in range(len(data))]


# 推荐硅基流动 https://siliconflow.cn
QDRANT_RERANK_URL = os.getenv('QDRANT_RERANK_URL', 'http://localhost:8081/v1/rerank')
QDRANT_RERANK_KEY = 'Bearer ' + os.getenv('QDRANT_RERANK_KEY', QDRANT_EMBD_KEY)
QDRANT_RERANK_MODEL = os.getenv('QDRANT_RERANK_MODEL', 'BAAI/bge-reranker-v2-m3')


def rerank(documents_: list, query_: str, top_n_: int = 3):
    response_content: str = post_json(url=QDRANT_RERANK_URL,
                                      data={
                                          'model': QDRANT_RERANK_MODEL,
                                          "top_n": top_n_,
                                          'documents': documents_,
                                          "query": query_
                                      },
                                      headers={'Authorization': QDRANT_RERANK_KEY})
    response: dict = json.loads(response_content)
    if type(response) is not dict or not response:
        print(response_content)
        return
    data: list = response.get('results', [])
    if not data:
        print(response_content)
        return
    res = []
    for chunk in data:
        res.append((documents_[chunk['index']], chunk['relevance_score']))
    res.sort(key=lambda x: x[1], reverse=True)
    return res


reg_q = re.compile(r'''['"“”‘’「」『』]''')
reg_e = re.compile(r'''[?!。？！]''')


def readChunks(filePath):
    with (gzip.open(filePath, 'rt', encoding='utf-8')
          if filePath.endswith('.gz') else open(filePath, encoding='utf-8') as f):
        retn = []
        cache = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = reg_q.sub(' ', line) + '\n'  # 删除引号
            if len(cache) + len(line) < 384:
                cache += line
                continue
            if not bool(reg_e.findall(line)):
                cache += line
                retn.append(cache.strip())
                cache = ''
                continue
            i = 1
            s = 0
            while i <= len(line):
                if len(cache) + (i - s) < 384:  # 每 384 切一行
                    i = (384 - len(cache)) + s
                    if i > len(line):
                        break
                    cache += line[s:i]
                    s = i
                if line[i - 1] in ('?', '!', '。', '？', '！'):
                    cache += line[s:i]
                    s = i
                    retn.append(cache.strip())
                    cache = ''
                i += 1
            if len(line) > s:
                cache += line[s:]

    cache = cache.strip()
    if cache:
        retn.append(cache)
    return retn


QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333') + '/collections/%s/points/search'
QDRANT_KEY = os.getenv('QDRANT_KEY', '')


def qdrant(vector_: list, collection_name_: str = 'my_collection', limit_: int = 5):
    response_content: str = post_json(url=QDRANT_URL % collection_name_,
                                      data={
                                          'vector': vector_,
                                          "limit": limit_,
                                          'with_payload': True
                                      },
                                      headers={'api-key': QDRANT_KEY})
    response: dict = json.loads(response_content)
    if type(response) is not dict or not response:
        print(response_content)
        return
    data: list = response.get('result', [])
    if not data:
        print(response_content)
        return
    return [res['payload']['text'] for res in data]


if __name__ == '__main__':
    print(QDRANT_EMBD_URL)
    print(QDRANT_EMBD_KEY)
    testQ = '机器人限拥令是什么'
    test1 = embd([testQ])
    print(QDRANT_RERANK_URL)
    print(QDRANT_RERANK_KEY)
    # test2 = rerank([
    #     "hi",
    #     "it is a bear",
    #     "world",
    #     "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species "
    #     "endemic to China."
    # ], "What is panda?")
    print(QDRANT_URL % 'my_collection')
    print(QDRANT_KEY)
    test3 = qdrant(test1[0][1][:256])
    test2 = rerank(test3, testQ)
    print(('\n' + '=' * 50 + '\n').join(x[0] for x in test2))
