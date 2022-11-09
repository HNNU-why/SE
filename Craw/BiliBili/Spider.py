import requests
import json
import math


class Spider:

    def __init__(self, headers):
        self.headers = headers

    def getVideo(self, search, maxVideo):
        videoList = []
        print('正在获取视频列表.......')
        for i in range(maxVideo):
            # noinspection PyStringFormat
            video_url = 'https://api.bilibili.com/x/web-interface/search/all/v2?keyword={}&page={' \
                        '}&duration=&tids_1=&tids_2=&__refresh__=true&_extra=&highlight=1&single_column=0&jsonp=jsonp' \
                        ''.format(search, i + 1)
            resp = requests.get(video_url, headers=self.headers)
            data = json.loads(resp.text)
            try:
                video_data = data['data']['result'][10]['data']
                for d in video_data:
                    videoList.append(d['id'])
            except KeyError:
                print('KeyError')
        print('视频列表获取成功！')
        return videoList

    # 获取评论页数
    def getReplyPageNum(self, oid):
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

    def getComment(self, video, maxPage):
        messageList = []
        print("正在从视频", video, " 获取评论")
        for page in range(1, self.getReplyPageNum(video) + 1):
            url = 'https://api.bilibili.com/x/v2/reply/main?mode=3&next={}&oid={}&plat=1&type=1'.format(page, video)
            resp = requests.get(url, headers=self.headers)
            try:
                replies = resp.json()['data']['replies']
                for reply in replies:
                    comment = {'message': reply['content']['message'], 'time': reply['reply_control']['time_desc']}
                    messageList.append(comment)
            except KeyError:
                print(KeyError)
            except TypeError:
                print(TypeError)
            # 控制每个视频爬取的评论页数
            if page > maxPage:
                break
        print("评论获取成功！")
        return messageList

    def saveComment(self, data, name):
        print("正在保存评论。。。。")
        with open('../Data/{}.json'.format(name), 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False)
            fp.close()
        print("评论保存成功")

    def readCommentFromFile(self, name):
        print("正在读取评论")
        jList = []
        with open('../Data/{}.json'.format(name), 'r', encoding='UTF-8') as fp:
            for line in fp.readlines():
                dic = json.loads(line)
                jList.append(dic)
            fp.close()
        print("评论读取成功")
        return jList[0]
