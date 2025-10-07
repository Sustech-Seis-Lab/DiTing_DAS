import numpy as np
import requests

# server_ip为运行服务的机器ip,
# server_port 为 server_conf.yml 中配置的端口号
server_ip = "10.16.47.196"
server_port = "1080"

# x 应该具有形状 (n,3,10000), 表示 n 条 3 分量波形, 每条波形有 10000 个点
# 第一分量为 Z, 第二分量为 N, 第三分量为 E.
x = np.random.rand(1, 3, 10000)

# 在向 api 提交数据的时候,需要将数组转换成 list,
# 并封装到字典里, 字典的 key 为 'array_data'
data = {'array_data': x.tolist()}

# 通过对应的 api 接口调用 DiTing 模型进行计算
# 接口统一为 inference, 根据配置文件确定不同的任务类型
output = requests.post(f'http://{server_ip}:{server_port}/inference/', json=data)

# 打印结果
print(output.json()['result']) 