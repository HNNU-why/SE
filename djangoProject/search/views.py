from pathlib import Path

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from Craw.BiliBili.Spider import Spider
import os.path


def craw(keyWord):
    headers = {
        'accept': 'application/json, text/plain, */*',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.42',
        'cookie': "buvid3=341E375F-63BF-D0E6-2D66-4F81DFBD059357749infoc; b_nut=1667868357; i-wanna-go-back=-1; "
                  "_uuid=7911D9C6-A1C5-2795-8E95-59E12555CDCF57926infoc; "
                  "buvid4=D2EB1F5B-17FA-0CB8-8C66-C1F8DB326AF058385-022110808-OrIQ3bLaNqvw1dl9ZT6Nkg%3D%3D; "
                  "nostalgia_conf=-1; rpdid=|(J|)RJYYRkl0J'uYY~)Y)J~m; hit-new-style-dyn=0; hit-dyn-v2=1; "
                  "share_source_origin=COPY; fingerprint=9d95cddb8278d5b55fb43d6fed67a6b4; buvid_fp_plain=undefined; "
                  "SESSDATA=9151d446%2C1683420470%2C92582%2Ab1; bili_jct=dc81f334ec7c4c553398fbe35f2d7933; "
                  "DedeUserID=399834015; DedeUserID__ckMd5=6b331073964aa2e6; sid=5nkyr2i3; "
                  "buvid_fp=9d95cddb8278d5b55fb43d6fed67a6b4; PVID=1; b_ut=5; bsource=search_bing; innersign=1; "
                  "CURRENT_QUALITY=64; bp_video_offset_399834015=726024278369108100; b_lsid=827B4FFF_18455E82BB3; "
                  "CURRENT_FNVAL=16",
        'Connection': 'keep-alive',
    }
    spider = Spider(headers)
    comment = '../Data/{}.json'.format(keyWord)
    if os.path.exists(comment):
        messageList = spider.readCommentFromFile(keyWord)
    else:
        messageList = []
        videoList = spider.getVideo(keyWord, 5)                     # 取前5页视频
        for video in videoList:                                     # 取每个视频前20页评论
            messageList.append(spider.getComment(video, 20))
        spider.saveComment(messageList, keyWord)
    return messageList


def search(request):
    if request.method == 'POST':
        keyword = request.POST.get('keyWord')
    print(keyword)
    messageList = craw(keyword)
    return render(request, 'show.html', {"videoList": messageList})


def index(request):
    return render(request, 'search.html')
