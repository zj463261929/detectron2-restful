#coding=utf8
'''
random.randint(a, b):用于生成一个指定范围内的整数。
其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b

random.choice(sequence)：从序列中获取一个随机元素
参数sequence表示一个有序类型（列表，元组，字符串）

'''
#import httplib
import time,json
import threading
from random import randint,choice 
import os
import requests
from threading import Timer
#创建请求函数
def postRequest():
    '''
    postJson={                     
                }
    #定义需要进行发送的数据
    postData=json.dumps(postJson)

    #定义一些文件头
    headerdata = {
      
        "content-type":"application/json",
         }
    
    #接口
    requrl ="/v1/query"
    
    #请求服务,例如：www.baidu.com
    hostServer=""
    #连接服务器       
    conn = httplib.HTTPConnection(hostServer)
    #发送请求       
    conn.request(method="POST",url=requrl,body=postData,headers=headerdata)
    
    #获取请求响应       
    response=conn.getresponse()
    #打印请求状态
    if response.status in range(200,300):
        print u"线程"+str(threadNum)+u"状态码："+str(response.status)       
    conn.close()   
    '''
    file = '20180921_50E54921F7C4_00008.jpg'
    imagepath_post = '/data/AI/zhangjing/detectron2/restful/img'
    image = os.path.join(imagepath_post,file)
    data={"data":image}
    my_json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    single_start = time.time()
    stime = time.localtime(single_start)
    poststartime=str(time.strftime("%Y%m%d%H%M%S",stime))
    data_secs = (single_start - int(single_start)) * 1000
    poststartime_sec = poststartime + '-'+str("%03d" % data_secs)
    print('poststart-time:{}'.format(poststartime_sec))
    s = requests
    #r = s.post('http://localhost:8080/user', headers=headers,data = my_json_data,)
    r = s.post('http://192.168.15.112:9527/user', headers=headers,data = my_json_data,)
    single_end = time.time() - single_start
    print ('cam:{}-time:{}'.format(0,single_end))
       

    
if __name__ == '__main__':
    for i in range(10000000): #7
        timer = Timer(2, postRequest) #0.5s=500ms
        timer.start()
        time.sleep(2)         
   
