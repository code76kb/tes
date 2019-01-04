# -*- coding: utf-8 -*-
import requests
import json
from jsoncomment import JsonComment


def translate(text,lanCode):
    

    url="https://api.multillect.com/translate/json/1.0/927"

    params={
    "method":"translate/api/translate",
    "from":"en",
    "to":lanCode,
    "text":text,
    "sig":"cb8f1f326b27f76f17fefe8355d738c5"
    }

    response = requests.get(url,params=params,headers={"content-type":"text","encoding":"utf8"})
    data = response.json()

    tre = data['result']['translated']
    #print 'tre :',tre
    return tre

en = open("/home/satyaprakash/en.json",'r')
hi  = open("/home/satyaprakash/tr.json",'w')

#file_string = '"en":'+str(en.read())

#print 'file string :\n',file_string 
parser = JsonComment(json)
en_json = parser.loads(en.read())

en.close()

# print 'En :\n',en_json
for key in en_json:

    print '\n translateing in ::',key
    
    subObject = '\n\n"'+key+'"'+':{'

    for key1 in en_json[key]:
         tra = translate(en_json[key][key1],'tr')
         print '\n   translateing :',en_json[key][key1],' in tr ',tra 

         subObject = subObject+"\n"+ '"'+key1+'"' +":" +'"'+tra+'"' +',' 
    
    subObject = subObject + "},"
    # print 'subObject :',type(subObject)
    hi.write(subObject.encode('utf-8'))

hi.close()      

