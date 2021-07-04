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


def cosine_sim(count):
    df =  pd.read_excel('news.xlsx',engine='openpyxl')
    df = df.drop_duplicates(['title'], keep='first', ignore_index=True)
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
    #print(sim_scores)
    
    # 원하는 기사와 유사한 기사 인덱스를 이용하여 제목 출력
    movie_indices = [i[0] for i in sim_scores]
    result = df['title'].iloc[movie_indices]
    print(result)
    print(tuple(result.index))
    
    df.set_index('date', inplace = True)
    df.to_excel('news.xlsx')
    
    return tuple(result.index)

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.okt = Okt()
        # 불용어 불러오기
        self.stopwords = [line.rstrip('\n') for line in open('stopwords_korean2.txt', encoding = 'utf-8')]
    
    def text2sentences(self, text):
        #sentences = self.kkma.sentences(text)\
        sentences = kss.split_sentences(text)
        sentences = sentences[0:len(sentences)-2]
        
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
                
        return sentences
    
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence))
                    if noun not in self.stopwords and len(noun) > 1]))
                
        return nouns


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence
    
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}
    
    
class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
                
            A[:, id] *= -d
            A[id, id] = 1
            
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}

    
    
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        self.sentences = self.sent_tokenize.text2sentences(text)
            
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        #print(self.nouns)
        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        
        return summary
    
    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        
        keywords = []
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
            
        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
            
        return keywords


class newsCrawlerNaver:
    def __init__(self):
        self.titleList=[]
        self.contentsList=[]
        self.imageList=[]
        self.dateList=[]
    # 네이버 뉴스홈
    def mainCrawl(self):    
        # 정치=100 경제=101 사회=102 생활/문화=103 세계=104 IT/과학=105
        for category in range(100, 106):
            main_url = "https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1="+str(category)
            driver.get(main_url)
            
            # '헤드라인 더보기' 버튼이 있다면 누르기       
            self.showMore()
            driver.implicitly_wait(0.5)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')  
            
            # 헤드라인 가져오기
            self.subCrawl(soup,category)
        #driver.quit()
    
    # 더보기버튼 클릭
    def showMore(self):
        try:
            while True:
                print("더보기")
                driver.find_element_by_xpath('//*[@id="main_content"]/div/div[2]/div[2]/div/a').click()
                driver.implicitly_wait(0.5)
        except exceptions.ElementNotVisibleException:
            pass
        except Exception:
            pass
    
    

    # 헤드라인 뉴스 크롤링
    def subCrawl(self, soup,category):
        # 모든 헤드라인 뉴스 저장
        articles = soup.find_all('div', {'class': 'cluster_group _cluster_content'})
        
        for i in range(len(articles)):
            # 각 뉴스 본문에 있는 이미지와 이미지URL를 저장할 리스트
            
            company=""
            
            images=[]
            imagesURL="NO IMAGE"

            temp = articles[i].find_all('div', {'class': 'cluster_text'})[0]

            conURL = temp.a.get('href')
            html2 = session.get(conURL,headers=headers).content
            soup2 = BeautifulSoup(html2, 'html.parser')
            
            company = soup2.find('meta', {'property':'me2:category1'}).get('content')
            
            summary = soup2.find('strong', {'class':'media_end_summary'})
            if summary==None:
                summary=""
            else:
                summary=summary.text
            
            content = soup2.find('div', id= "articleBodyContents").text.replace("\n"," ").replace(str(summary),"").replace("\t"," ").replace("// flash 오류를 우회하기 위한 함수 추가 function _flash_removeCallback() {}"," ")
            title=soup2.find('h3',id="articleTitle").text
            
            # 기사 본문이 10문장이하라면 저장하지 않는다.
            if(len(kss.split_sentences(content)) <= 10):
                continue;

            date=soup2.find('span', {'class','t11'}).text

            images=soup2.find_all('span', {'class','end_photo_org'})
            
            for i in range(len(images)):
                imagesURL=(images[i].find("img")["src"])
                
            self.titleList.append(title)
            self.contentsList.append(content)
            self.dateList.append(date)
            
            self.saveToDB(str(title),str(content),str(imagesURL),str(date),str(category),str(company))
        # 엑셀 파일 읽기
        try:
            news_df = pd.read_excel('news.xlsx',engine='openpyxl')
        except:
            temp = pd.DataFrame(columns=('date','title','content','category'))
            temp.set_index('date', inplace=True)
            temp.to_excel('news.xlsx')
            news_df = pd.read_excel('news.xlsx',engine='openpyxl')
            
        news_df.set_index('date', inplace = True)
    
        # 크롤링한 기사 정보로 데이터프레임 생성
        crawl_df = pd.DataFrame({'title':self.titleList, 'content':self.contentsList, 'date':self.dateList})
        crawl_df['category'] = pd.Series([category for i in range(len(crawl_df.index))])
        crawl_df.set_index('date', inplace = True)
        
        news_df = pd.concat([news_df, crawl_df])
        
        # 엑셀 파일 쓰기
        news_df.to_excel('news.xlsx')            
        self.titleList=[]
        self.contentsList=[]
        self.imageList=[]
        self.dateList=[]


    def saveToDB(self,title,content,imagesURL,date,category,company):
        content=content.replace("'","")
        sum = TextRank(content)
        
        content=sum.summarize(7)
        count=1
        for i in content:
            if i=="":
                print('중지됨')
                return 3
            print(i)
            print(count)
            print("\n")
            count=count+1
        if len(content)<7:
            return 2
        title = title.replace("'","")

        # SQL문 실행
        sql = "USE TEST1"
        curs.execute(sql)
        '''
        CREATE TABLE NEWS8 (TITLE VARCHAR(200) NOT NULL UNIQUE,
        CONTENT1 VARCHAR(500) NOT NULL,
        CONTENT2 VARCHAR(500) NOT NULL,
        CONTENT3 VARCHAR(500) NOT NULL,
        CONTENT4 VARCHAR(500) NOT NULL,
        CONTENT5 VARCHAR(500) NOT NULL,
        CONTENT6 VARCHAR(500) NOT NULL,
        CONTENT7 VARCHAR(500) NOT NULL,
        COMPANY VARCHAR(20) NOT NULL,
        DATE VARCHAR(40) NOT NULL,
        CATEGORY VARCHAR(40),
        COUNT int NOT NULL AUTO_INCREMENT,
        IMAGE TEXT NOT NULL,
        CONSTRAINT PLAYER_PK PRIMARY KEY (COUNT));
        '''

        #sql3="insert into NEWS3(title,content,date,category,image) VALUES(" +title+ ',' +content+ ',' +date+ ',' +category+ ',' +imagesURL+ ");"
        sql3="""insert into NEWS1000(title,content1,content2,content3,content4,content5,content6,content7,company,date,category,image) VALUES('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s');"""%(title,content[0],content[1],content[2],content[3],content[4],content[5],content[6],company,date,category,imagesURL)
        try:
            curs.execute(sql3)
        except:
            pass
        conn.commit()
        print("db updated!")

        return 1

class newsCrawlerNate:
    def __init__(self):
        self.titleList=[]
        self.contentsList=[]
        self.imageList=[]
        self.dateList=[]
        
    # 네이트 뉴스홈
    def mainCrawl(self):    
        
        for category in range(200, 601,100):
            main_url = "https://news.nate.com/section?mid=n0"+str(category)
            driver.get(main_url)
            
            driver.implicitly_wait(0.1)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')  
            
            # 헤드라인 가져오기
            self.subCrawl(soup,category)
        driver.quit()
    

    # 헤드라인 뉴스 크롤링
    def subCrawl(self, soup, category):
                # 모든 헤드라인 뉴스 저장
        articles = soup.find_all('div', {'class': 'mlt01'})

        for i in range(len(articles)):
            # 각 뉴스 본문에 있는 이미지와 이미지URL를 저장할 리스트
            
            company=""
            
            imagesURL="NO IMAGE"

            conURL = "https:" + articles[i].a.get('href')

            html2 = session.get(conURL,headers=headers).content

            soup2 = BeautifulSoup(html2, 'html.parser')

            company = "test"
            
            imgSummary=soup2.find_all('span', {'class':'sub_tit'})#.text
                        
            summary = soup2.find('strong', {'class':'media_end_summary'})
            if summary==None:
                summary=""
            else:
                summary=summary.text
            
            contentTemp=""
            try:
                contentTemp=soup2.find('div', id= "realArtcContents").find('dl').getText()
            except:
                try:
                    contentTemp=soup2.find('div', id= "realArtcContents").find('ul').getText()
                except:
                    
                    pass
                
            contentTemp2=soup2.find('div', id= "realArtcContents").find('script').getText()
            contentTemp3=soup2.find('div', id= "realArtcContents").find_all('a')
            
            content = soup2.find('div', id= "realArtcContents").getText()
            
            for i in range(len(imgSummary)):
                content.replace(imgSummary[i].text,"")
            for i in range(len(contentTemp3)):
                content.replace(contentTemp3[i].text,"")
                

            content = content.replace(contentTemp," ").replace(contentTemp2," ").replace("\n"," ").replace("\t"," ")
            title = soup2.find('meta', {'property':'og:title'}).get('content')
            images=soup2.find('meta', {'property':'og:image'})
            date=soup2.find('span', {'class','firstDate'}).find('em').getText()
            if images!=None:
                imagesURL=images.get('content')


            # 기사 본문이 10문장이하라면 저장하지 않는다.
            if(len(kss.split_sentences(content)) <= 10):
                continue;
                
            self.titleList.append(title)
            self.contentsList.append(content)
            self.dateList.append(date)

            self.saveToDB(str(title),str(content),str(imagesURL),str(date),str(category),str(company))

        # 엑셀 파일 읽기
        try:
            news_df = pd.read_excel('news.xlsx',engine='openpyxl')
        except:
            temp = pd.DataFrame(columns=('date','title','content','category'))
            temp.set_index('date', inplace=True)
            temp.to_excel('news.xlsx')
            news_df = pd.read_excel('news.xlsx',engine='openpyxl')
        news_df.set_index('date', inplace = True)
    
        # 크롤링한 기사 정보로 데이터프레임 생성
        crawl_df = pd.DataFrame({'title':self.titleList, 'content':self.contentsList, 'date':self.dateList})
        crawl_df['category'] = pd.Series([category for i in range(len(crawl_df.index))])
        crawl_df.set_index('date', inplace = True)
        
        news_df = pd.concat([news_df, crawl_df])
        
        # 엑셀 파일 쓰기
        news_df.to_excel('news.xlsx')            
        self.titleList=[]
        self.contentsList=[]
        self.imageList=[]
        self.dateList=[]
            
            
            
    def saveToDB(self,title,content,imagesURL,date,category,company):
        content=content.replace("'","")
        sum = TextRank(content)
        
        content=sum.summarize(7)
        count=1
        for i in content:
            if i=="":
                print('중지됨')
                return 3
            print(i)
            print(count)
            print("\n")
            count=count+1
        if len(content)<7:
            return 2
        title = title.replace("'","")

        # SQL문 실행
        sql = "USE TEST1"
        curs.execute(sql)
        '''
        CREATE TABLE NEWS8 (TITLE VARCHAR(200) NOT NULL UNIQUE,
        CONTENT1 VARCHAR(500) NOT NULL,
        CONTENT2 VARCHAR(500) NOT NULL,
        CONTENT3 VARCHAR(500) NOT NULL,
        CONTENT4 VARCHAR(500) NOT NULL,
        CONTENT5 VARCHAR(500) NOT NULL,
        CONTENT6 VARCHAR(500) NOT NULL,
        CONTENT7 VARCHAR(500) NOT NULL,
        COMPANY VARCHAR(20) NOT NULL,
        DATE VARCHAR(40) NOT NULL,
        CATEGORY VARCHAR(40),
        COUNT int NOT NULL AUTO_INCREMENT,
        IMAGE TEXT NOT NULL,
        CONSTRAINT PLAYER_PK PRIMARY KEY (COUNT));
        '''

        
        sql3="""insert into NEWS1000(title,content1,content2,content3,content4,content5,content6,content7,company,date,category,image) VALUES('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s');"""%(title,content[0],content[1],content[2],content[3],content[4],content[5],content[6],company,date,category,imagesURL)
        try:
            curs.execute(sql3)
        except:
            pass
        conn.commit()
        print("db updated!")

        return 1
    
    
    
conn = pymysql.connect(host='newdb.c7p2ncpgik7h.ap-northeast-2.rds.amazonaws.com', user='admin', password='1dlckdals!',
                       db='TEST1', charset='utf8')
curs = conn.cursor()
session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}


chrome_options=webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(r"/home/capston/chromedriver",chrome_options=chrome_options)
#driver = webdriver.Chrome(r"C:\Users\LCM\Downloads\chromedriver_win32 (4)\chromedriver.exe")


crawlNaver=newsCrawlerNaver() 
crawlNaver.mainCrawl()
crawlNate=newsCrawlerNate() 
crawlNate.mainCrawl()



conn.commit()
curs.close()
print('done')
