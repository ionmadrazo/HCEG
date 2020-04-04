import requests
import json
import time
class panlexAPI:
    def __init__(self,langCodeTranslator=None):
        if langCodeTranslator is not None:
            self.langCodeTranslator=langCodeTranslator
        else:
            self.langCodeTranslator=None

    def getLanguageUID(self, languageCode ):

        if self.langCodeTranslator is None:
            assert len(languageCode) == 3, "languageCode must contain 3 characters of a languageCodeTranslator objects needs to be provided during init"
            return "{}-000".format(languageCode)
        else:
            if len(languageCode)==2 :
                return "{}-000".format(self.langCodeTranslator.transform23(code=languageCode))
            else:
                return "{}-000".format(languageCode)

    def translate(self, langFrom,langTo, word):
        exprId = self.getExprId(langFrom,word)
        if exprId is None:
            return None
        langToUID = self.getLanguageUID(langTo)
        url = "http://api.panlex.org/v2/expr"
        data = { "uid": langToUID, "trans_expr": exprId , "indent": True, "include":["trans_quality"], "sort":"trans_quality desc"}
        response = requests.post(url, data=json.dumps(data))
        responseDict= response.json()
        if "result" in responseDict and len(responseDict["result"])>0 and "txt" in responseDict["result"][0]:
            return responseDict["result"][0]["txt"]
        if "code" in responseDict and responseDict["code"]== 'TooManyRequests':
            time.sleep(5)
            print("API limit reached, waiting 5 seconds...")
            return self.translate(langFrom,langTo, word)


        return None

    def translateMultiple(self, langFrom,langTo, words=[]):
        txt2ExprId = self.getMultipleExprId(langFrom,words)
        if txt2ExprId is None:
            return None
        exprId2txt =  {txt2ExprId[k]: k for k in txt2ExprId}
        langToUID = self.getLanguageUID(langTo)
        
        url = "http://api.panlex.org/v2/expr"
        data = { "uid": langToUID, "trans_expr": [txt2ExprId[key] for key in txt2ExprId] , "indent": True, "include":["trans_quality"], "sort":"trans_quality desc"}
        response = requests.post(url, data=json.dumps(data))
        responseDict= response.json()
        #print(responseDict)
        src2tgt= {}
        if "result" in responseDict and len(responseDict["result"])>0 :
            for result in responseDict["result"]:
                transExpr= result["trans_expr"]
                tgtToken = result["txt"]
                srcToken = exprId2txt[transExpr]
                if srcToken not in src2tgt:
                    src2tgt[srcToken]=tgtToken
                #txt2ExprId[result["txt"]]=result["id"]
            return src2tgt
        return None

    def getExprId(self, lang, expr):
        langUID=self.getLanguageUID(lang)
        url = 'http://api.panlex.org/v2/expr'
        data = { "uid": langUID, "txt": expr }
        response = requests.post(url, data=json.dumps(data))
        responseDict= response.json()
        if "result" in responseDict and len(responseDict["result"])>0 and "id" in responseDict["result"][0]:
            return responseDict["result"][0]["id"]
        if "code" in responseDict and responseDict["code"]== 'TooManyRequests':
            time.sleep(5)
            print("API limit reached, waiting 5 seconds...")
            return self.getExprId(lang, expr)
        return None


    def getMultipleExprId(self, lang, expr=[]):
        langUID=self.getLanguageUID(lang)
        url = 'http://api.panlex.org/v2/expr'
        data = { "uid": langUID, "txt": expr }
        response = requests.post(url, data=json.dumps(data))
        responseDict= response.json()
        txt2ExprId = {}
        if "result" in responseDict and len(responseDict["result"])>0 :
            for result in responseDict["result"]:
                txt2ExprId[result["txt"]]=result["id"]

            return txt2ExprId
        return None
