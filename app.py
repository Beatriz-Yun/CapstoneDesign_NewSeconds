import urllib.request
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
import pymysql
from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import kss
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask import Flask, request # request import 추가
import json
def cosine_sim(count):
    news_df =  pd.read_excel('news.xlsx', engine='openpyxl')
    df = news_df.drop_duplicates(['title'], keep='first', ignore_index=True)
    # 모든 단어들의 빈도에 대하여 유사도를 계산하면 값이 너무 작게 나와서
    # max_features옵션으로 Tf-Idf의 크기를 줄인 다음 코사인 유사도를 계산함.
    tfidf = TfidfVectorizer(max_features=100)
    # max_feature는 tf-idf vector의 최대 feature를 설정해주는 파라미터입니다.
    # 머신러닝에서 feature란, 테이블의 컬럼에 해당하는 개념입니다. 또한 행렬의 열에 해당하는 것이기도 합니다.
    # TF-IDF 벡터는 단어사전의 인덱스만큼 feature를 부여받습니다.
    
    # doc: 기사 본문(문서)
    # tfidf_mat: 문서들을 벡터화한 
    doc = list(df['content'])
    tfidf_mat = tfidf.fit_transform(doc).toarray()
    #print(type(tfidf_mat))
    #print(tfidf_mat)
    #print(tfidf_mat.shape)
    
    # 소수점 4자리까지 반올림
    sim = np.round(cosine_similarity(tfidf_mat, tfidf_mat),4)
    #print(sim)
    #print(type(sim))
    #print(sim.shape)
    
    #print(sim)
    
    sim_scores = list(enumerate(sim[count]))
    #print(sim_scores)
    
    # 유사도가 높은 순서대로 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 상위 인덱스와 유사도 추출
    sim_scores = sim_scores[1:6]
    print(sim_scores)
    
    # 원하는 기사와 유사한 기사 인덱스를 이용하여 제목 출력
    movie_indices = [i[0] for i in sim_scores]
    result = df['title'].iloc[movie_indices]
    print(result)
    print(tuple(result.index))
    
    df.set_index('date', inplace = True)
    df.to_excel('news.xlsx')
    
    return list((tuple(result.index)))

app = Flask (__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def root():
    parameter_dict = request.args.to_dict()
    if len(parameter_dict) == 0:
        return 'No paramete1223r4'

    parameters = ''
    result_list=[]
    for key in parameter_dict.keys():
        result_list.extend(cosine_sim(int(request.args[key])))
        print(request.args[key])
    parameters=" ".join([str(_) for _ in result_list])
    return parameters


@app.route('/update')
def root2():
    # EX) 127.0.0.1:5000/update?id=abc&newsid=4
    parameter_dict = request.args.to_dict()
    if len(parameter_dict) == 0:
        return '오류'
    a=list()
    for key in parameter_dict.keys():
        a.append(request.args[key])

    SQL='SELECT * FROM USER1 WHERE USERID="'+a[0]+'"'
    curs.execute(SQL)
    temp=curs.fetchall()
    temp=temp[0][2]

    sent=''
    temp2=list()
    for i in range(len(temp)):
        if i==len(temp)-1:
            sent += temp[i]
            temp2.append(int(sent))

        elif temp[i]==',':
            temp2.append(int(sent))
            sent=''

        else:
            sent+=temp[i]

    if int(a[1]) not in temp2:
        temp2.append(int(a[1]))
        sentence=''
        for i in range(len(temp2)):
            sentence+=str(temp2[i])+','
        sentence=sentence[0:-1]

        SQL="UPDATE USER1 SET PREFER_NEWSID='%s' WHERE USERID='%s'"%(sentence,a[0])
        curs.execute(SQL)
        conn.commit()
        return "업데이트 성공"
    else:
        return "업데이트 실패(이미 선택했던 뉴스일 수 있음)"



@app.route('/signin')
def index2():
    # EX) 127.0.0.1:5000/signin?id=abc&pw=123&news=1,2,3
    parameter_dict = request.args.to_dict()
    if len(parameter_dict) == 0:
        return 'No parameter select'

    a=list()
    for key in parameter_dict.keys():
        a.append(request.args[key])

    SQL='SELECT * FROM USER1 WHERE USERID="'+a[0]+'"'
    temp=curs.execute(SQL)

    if temp==1: # 이미 존재하는 아이디
        return "실패"
    else:
        SQL="INSERT INTO USER1(USERID,USERPW,PREFER_NEWSID) VALUES('%s','%s','%s');"%(a[0],a[1],a[2])
        curs.execute(SQL)
        conn.commit()
    return "성공"


@app.route('/login')
def asdf():
    # EX) 127.0.0.1:5000/login?id=abc&pw=123
    parameter_dict = request.args.to_dict()
    if len(parameter_dict) == 0:
        return '오류'

    a=list()
    for key in parameter_dict.keys():
        a.append(request.args[key])

    SQL='SELECT * FROM USER1 WHERE USERID="'+a[0]+'"'
    curs.execute(SQL)
    temp=curs.fetchall()

    if temp[0][0]==a[0] and temp[0][1]==a[1]:
        return "로그인 성공"
    else:
        return "실패"



@app.route('/select')
def root3():

    # user1 테이블의 해당 유저 userid를 가져온다.
    # EX) 127.0.0.1:5000/select?param=abc

    parameter_dict = request.args.to_dict()
    if len(parameter_dict) == 0:
        return 'No parameter select'


    a="default"
    for key in parameter_dict.keys():
        a=(request.args[key])

    curs.execute('SELECT * FROM USER1 WHERE USERID="%s";'%(a))
    bbb=curs.fetchall()
    aaa=bbb[0][2]
    #aaa=aaa.replace(',','c')

    sent=''
    temp2=list()
    for i in range(len(aaa)):

        if i==len(aaa)-1:
            sent += aaa[i]
            temp2.append(sent)

        elif aaa[i]==',':
            temp2.append(sent)
            sent=''

        else:
            sent+=aaa[i]

    result=list()
    for i in range(len(temp2)):
        result.append(cosine_sim(int(temp2[i])))

    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j]+=2

    curs.execute('SELECT * FROM NEWS88 WHERE COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s" OR COUNT="%s";'% (result[0][0],result[0][1],result[0][2],result[0][3],result[0][4],result[1][0],result[1][1],result[1][2],result[1][3],result[1][4],result[2][0],result[2][1],result[2][2],result[2][3],result[2][4]) )


    data = json.dumps(curs.fetchall(),ensure_ascii=False)
    return data

    
if __name__ == "__main__":
    conn = pymysql.connect(host='newdb.c7p2ncpgik7h.ap-northeast-2.rds.amazonaws.com', user='admin', password='1dlckdals!',
                       db='TEST1', charset='utf8')
    curs = conn.cursor()
    session = requests.Session()
    app.run(host='0.0.0.0',port=5000)



