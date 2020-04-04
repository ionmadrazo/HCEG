class languageCodeTranslator:


    def __init__(self,filename="data/languageCodes/ISO-639-2_utf-8.txt"):
        self.trans23={}
        self.trans32={}
        self.trans2desc={}
        self.trans3desc={}
        with open(filename, "r") as f:
            for line in f:
                code3, code3alt, code2, desc1, desc2 =line.strip().split("|")
                if len(code2.strip())>0:
                    if len(code3alt.strip())>0:
                        self.trans23[code2]=code3alt
                    else:
                        self.trans23[code2]=code3
                    self.trans2desc[code2]=desc1

                if len(code3alt.strip())>0:
                    self.trans32[code3alt]=code2
                    self.trans3desc[code3alt]=desc1
                else:
                    self.trans32[code3]=code2
                    self.trans3desc[code3]=desc1



    def transform23(self, code):
        if code=="no":
            return "nno"
        if len(code)==3:
            return code
        return self.trans23[code]

    def transform32(self, code):
        if len(code)==2:
            return code
        return self.trans32[code]

    def getDescription(self, code):
        codeLen= len(code.strip())
        assert codeLen==3 or codeLen==2
        if codeLen==3:
            return self.trans3desc[code]
        else:
            return self.trans2desc[code]
