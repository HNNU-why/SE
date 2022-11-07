import requests
from bs4 import BeautifulSoup
import json
import math

"""
    Bilibili简易评论抓取
"""

headers = {
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.42',
    'cookie': "buvid3=1BE9BF5E-174E-E36F-638F-A83EBFE3923B73812infoc; buvid_fp=1BE9BF5E-174E-E36F-638F-A83EBFE3923B73812infoc; blackside_state=0; rpdid=|(u)Rm|uR|)l0J'uYRukmJ~RJ; LIVE_BUVID=AUTO1016419942077939; DedeUserID=1128547717; DedeUserID__ckMd5=6e2c06fc4d4a080a; b_ut=5; fingerprint_s=a1a842bc7f2ecb138fcd0ee9ebd6c264; i-wanna-go-back=-1; nostalgia_conf=-1; is-2022-channel=1; fingerprint3=da72afc7e244beb108e5842d6c8c8ccb; go_old_video=-1; i-wanna-go-feeds=-1; CURRENT_BLACKGAP=0; CURRENT_QUALITY=80; buvid4=null; fingerprint=aae73dad5de59138dcd2680c4d451960; buvid_fp_plain=undefined; b_nut=1666187479; SESSDATA=4df18d94%2C1683255514%2C2be5f%2Ab1; bili_jct=a2c88699d75f8024401a4deae6d82e76; sid=8vjk7l6x; bp_video_offset_1128547717=725453644720242800; innersign=1; CURRENT_FNVAL=4048",
    'Connection': 'keep-alive',
	'TE': 'Trailers'    
}

videoList = []

def getVideo():
    search = input("输入搜索关键字：")
    print('正在获取视频列表.......')
    for i in range(1):
        video_url = 'https://api.bilibili.com/x/web-interface/search/all/v2?keyword={}&page={}&duration=&tids_1=&tids_2=&__refresh__=true&_extra=&highlight=1&single_column=0&jsonp=jsonp'.format(search,i+1)
        resp = requests.get(video_url,headers=headers)
        data = json.loads(resp.text)
        try:
            video_data = data['data']['result'][10]['data']
            for d in video_data:
                videoList.append(d['id'])
        except KeyError:
            print('KeyError')
    print('视频列表获取成功！')

        

        # bs = BeautifulSoup(resp.text,"html.parser")
        # video_aid = bs.find_all()



def getReplyPageNum(oid):
    url="https://api.bilibili.com/x/v2/reply?&jsonp=jsonp&pn=1"+"&type=1&oid="+str(oid)+"&sort=2"
    respond=requests.get(url)
    res_dirct=json.loads(respond.text)
    replyPageNum = 1
    try:
        replyNum=int(res_dirct['data']['page']['acount'])
        replyPageCount=int(res_dirct['data']['page']['count'])
        replyPageSize=int(res_dirct['data']['page']['size'])
        replyPageNum=math.ceil(replyPageCount/replyPageSize)
    except KeyError:
        print("KeyError")
    return  replyPageNum

# def getAid(bvid):
#     url = "http://api.bilibili.com/x/web-interface/view?bvid="+str(bvid)
#     response = requests.get(url)
#     dirt=json.loads(response.text)
#     aid=dirt['data']['aid']
#     return aid

if __name__ == "__main__":
    getVideo()
    for video in videoList:
        messageList = []
        for page in range(1,getReplyPageNum(video)+1):
            url = 'https://api.bilibili.com/x/v2/reply/main?mode=3&next={}&oid={}&plat=1&type=1'.format(page,video)
            resp = requests.get(url, headers=headers)
            try:
                replies = resp.json()['data']['replies']
                for reply in replies:
                    messageList.append(reply['content']['message'])
            except KeyError:
                print(KeyError)
            # 控制每个视频爬取的评论页数
            if page > 30:
                break
        with open('comments.json','a',encoding='utf-8') as fp:
            json.dump(messageList,fp,ensure_ascii=False)
            fp.close()

