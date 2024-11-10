import os
import json
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
QDRANT_EMBD_KEY = os.getenv('QDRANT_EMBD_KEY', 'no-key')


def embd(input_: list):
    response_content: str = post_json(url=QDRANT_EMBD_URL,
                                      data={
                                          'model': 'GPT-4o',
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


if __name__ == '__main__':
    print(QDRANT_EMBD_URL)
    print(QDRANT_EMBD_KEY)
    test = embd(['你好'])

