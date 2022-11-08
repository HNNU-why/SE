import requests
import json
import math
import os

"""
    Bilibili简易评论抓取
"""
# 请填写自己的cookie
headers = {
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 '
                  'Safari/537.36 Edg/105.0.1343.42',
    'cookie': "buvid3=341E375F-63BF-D0E6-2D66-4F81DFBD059357749infoc; b_nut=1667868357; i-wanna-go-back=-1; "
              "b_lsid=C310C3D4B_18454B3751F; _uuid=7911D9C6-A1C5-2795-8E95-59E12555CDCF57926infoc; "
              "buvid4=D2EB1F5B-17FA-0CB8-8C66-C1F8DB326AF058385-022110808-OrIQ3bLaNqvw1dl9ZT6Nkg%3D%3D; "
              "nostalgia_conf=-1; rpdid=|(J|)RJYYRkl0J'uYY~)Y)J~m; hit-new-style-dyn=0; hit-dyn-v2=1; "
              "share_source_origin=COPY; fingerprint=9d95cddb8278d5b55fb43d6fed67a6b4; buvid_fp_plain=undefined; "
              "SESSDATA=9151d446%2C1683420470%2C92582%2Ab1; bili_jct=dc81f334ec7c4c553398fbe35f2d7933; "
              "DedeUserID=399834015; DedeUserID__ckMd5=6b331073964aa2e6; sid=5nkyr2i3; "
              "buvid_fp=9d95cddb8278d5b55fb43d6fed67a6b4; bp_video_offset_399834015=725974138913554400; PVID=1; "
              "CURRENT_FNVAL=4048; innersign=0; b_ut=5; bsource=search_bing",
    'Connection': 'keep-alive',
}


# 获取视频列表
def getVideo():
    videoList = []
    search = input("输入搜索关键字：")
    print('正在获取视频列表.......')
    for i in range(1):
        video_url = 'https://api.bilibili.com/x/web-interface/search/all/v2?keyword={}&page={' \
                    '}&duration=&tids_1=&tids_2=&__refresh__=true&_extra=&highlight=1&single_column=0&jsonp=jsonp' \
                    ''.format(search, i + 1)
        resp = requests.get(video_url, headers=headers)
        print(resp)
        data = json.loads(resp.text)
        try:
            video_data = data['data']['result'][10]['data']
            for d in video_data:
                videoList.append(d['id'])
        except KeyError:
            print('KeyError')
    print('视频列表获取成功！')
    return videoList, search


# 获取评论页数
def getReplyPageNum(oid):
    url = "https://api.bilibili.com/x/v2/reply?&jsonp=jsonp&pn=1" + "&type=1&oid=" + str(oid) + "&sort=2"
    respond = requests.get(url)
    res_dirct = json.loads(respond.text)
    replyPageNum = 1
    try:
        replyNum = int(res_dirct['data']['page']['acount'])
        replyPageCount = int(res_dirct['data']['page']['count'])
        replyPageSize = int(res_dirct['data']['page']['size'])
        replyPageNum = math.ceil(replyPageCount / replyPageSize)
    except KeyError:
        print("KeyError")
    return replyPageNum


# Bv号转化为av号
# def getAid(bvid):
#     url = "http://api.bilibili.com/x/web-interface/view?bvid="+str(bvid)
#     response = requests.get(url)
#     dirt=json.loads(response.text)
#     aid=dirt['data']['aid']
#     return aid


def getComment(param):
    videoList = [50756086]
    for video in videoList:
        messageList = []

        print("正在从", video, "获取评论")
        for page in range(1, getReplyPageNum(video) + 1):
            url = 'https://api.bilibili.com/x/v2/reply/main?mode=3&next={}&oid={}&plat=1&type=1'.format(page, video)
            resp = requests.get(url, headers=headers)
            try:
                replies = resp.json()['data']['replies']
                for reply in replies:
                    # messageList.append(reply['content']['message'])
                    comment = {'message': reply['content']['message'], 'time': reply['reply_control']['time_desc']}
                    messageList.append(comment)
            except KeyError:
                print(KeyError)
            # 控制每个视频爬取的评论页数
            if page > 30:
                break
        print(messageList)
        # saveComment(messageList, param[1])


def saveComment(messageList, search):
    print("正在保存评论。。。。")
    path = '{}.json'.format(search)
    # if not os.path.exists(path):
    #     os.mknod(path)                            # Linux环境下
    with open(path, 'a', encoding='utf-8') as fp:
        json.dump(messageList, fp, ensure_ascii=False)
        fp.close()
    print("评论保存成功")


if __name__ == "__main__":
    getComment(getVideo())
