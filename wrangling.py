import pandas as pd
import numpy as np
import re
from fuzzysearch import find_near_matches

def read_data(path):
    raw = pd.read_csv("news.csv")
    topics = ["Business", "investasi", "market", 
          "Tech", "teknologi", "stocksetup", 
          "Business", "Finance", "IT and Telco", "TEKNO",
          "finansial", "investasi", "keuangan", "telkomindonesia", "Market"]
    return raw.loc[raw["topic"].isin(topics), :]

def get_stock_dict(path):
    stocks = pd.read_excel(path, usecols = ["Ticker", "Nama"], keep_default_na = False)
    stock_names = stocks.iloc[:,1]
    tickers = stocks.iloc[:,0]
    zip_iterator = zip(tickers, stock_names)
    stock_dict = dict(zip_iterator)
    stock_dict["TRUE"] = stock_dict.pop(True)
    return stock_dict

def search_ticker_title(text, stock_dict):
    res = []
    for names in stock_dict:
            test_title_full = find_near_matches(stock_dict[names], text, max_l_dist=1)
        #### Depan space or blkng non text character 
            pattern = r"\b{temp}\b".format(temp = names)
            test_title_ticker = re.search(pattern, text)
            if test_title_full:
                res.append(names)
            elif test_title_ticker:
                res.append(names) 
    return res

def find_alternative(text):
    alt_stock = {"BBCA": r"\bBCA\b",
             "BBRI": r"\bBRI\b(?! Syariah| Agroniaga)",
             "AGRO": r"\bBRI Agroniaga\b",
             "BBNI": r"\bBNI (?! Syariah)\b",
             "BRIS": r"\bBRI Syariah\b",
             "ANTM": r"\bAntam\b",
             "AASI": r"\bAstra\b(?! Otoparts| Graphia| Agro)",
             "TLKM": r"\bTelkom",
             "BMRI": r"\bMandiri\b",
             "SMRA": r"\bSumarecon\b",
             "BNII": r"\bMaybank\b",
             "PWON": r"\bPakuwon\b",
             "MNCN": r"\bMNCN\b",
             "UNVR": r"\bUnilever\b",
             "AKRA": r"\bAKR\b",
             "MAYA": r"\bMayapada\b",
             "INTP": r"\bIndocement\b",
             "BDMN": r"\bDanamon\b",
             "BNBR": r"\bBakrie and Brothers\b",
             "BANK": r"\bBank Aladin\b",
             "BBTN": r"\bBTN\b",
             "LIFE": r"\b(Sinarmas MSIG Life|Sinarmas Life)\b",
             "EXCL": r"\bXL\b",
             "BNGA": r"\bCIMB Niaga\b",
             "PNBN": r"\bPanin\b(?! Syariah)",
             "SMGR": r"\bSemen Gresik\b",
             "ACES": r"\bAce Hardware\b",
             "IMAS": r"\bIndomobil\b"}
    res = []
    for names in alt_stock:
            test_title_ticker = re.search(alt_stock[names], text)
            if test_title_ticker:
                res.append(names)
    return res


def translate_text_en(text, key):
    import os
    # Set environment variables
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    result = translate_client.translate(text, target_language= "en")
    
    return result["translatedText"]


### Manual Change
def clean_ticker(title, ticker):
    res = ticker.copy()
    if "BANK" in ticker:
        if not (re.search(r"\bbank aladin\b", title.lower()) or re.search(r"\bbank net\b", title.lower()) or re.search(r"-BANK-", title)):
            res.remove("BANK")     
                
    if ("WTON" in res and "WIKA" in res):
        if not(re.search(r"\bwijaya karya\b(?! beton| bangunan)", title.lower()) or re.search(r"\bwika\b(?! beton| bangunan)", title.lower())):
            res.remove("WIKA")
            
    if ("WEGE" in res and "WIKA" in res):
        if not(re.search(r"\bwijaya karya\b(?! beton| bangunan)", title.lower()) or re.search(r"\bwika\b(?! beton| bangunan)", title.lower())):
            res.remove("WIKA")
            
    if ("DPNS" in res and "DUTI" in res):
        if not(re.search(r"\bduta pertiwi\b(?! nusantara)", title.lower()) or re.search(r"\bDUTI\b", title)):
            res.remove("DUTI")
            
    if ("BRMS" in res and "BUMI" in res):
        if not(re.search(r"\bbumi resources\b(?! minerals| mineral)", title.lower()) or re.search(r"\bBUMI\b", title)):
            res.remove("BUMI")

    if ("BTPN" in res and "BBTN" in res):
        if not(re.search(r"\bbank btpn\b", title.lower()) or re.search(r"\bBTPN\b", title)):
            res.remove("BTPN")

    if ("PNBN" in res and "PNBS" in res):
        if not(re.search(r"\bbank panin\b(?! syariah)", title.lower()) or re.search(r"\bPNBN\b", title)):
            res.remove("PNBN")
            
    if ("BLUE" in res and "BIRD" in res):
        if (re.search(r"\bBLUE BIRD\b")):
            res.remove("BLUE")
    
    if "INDO" in res:
        res.remove("INDO")
    if "CITY" in res:
        res.remove("CITY")
    if "LABA" in res:
        res.remove("LABA")
    if "AKSI" in res:
        res.remove("AKSI")
    if "LAND" in res:
        res.remove("LAND")
    
    if "BINA" in res:
        res.remove("BIMA")
    if "CASH" in res:
        res.remove("CASH")
    if "AMAN" in res:
        res.remove("AMAN")
    if "BALI" in ticker:
        res.remove("BALI")
    
    if "CARE" in res:
        res.remove("CARE")
    if "DAYA" in res:
        res.remove("DAYA")
    if "BUDI" in res:
        res.remove("BUDI")
    if "BALI" in ticker:
        res.remove("BALI")
    if "FOOD" in res:
        res.remove("FOOD")
    if "LIFE" in res:
        res.remove("LIFE")
    if "AASI" in res:
        res.remove("AASI")
        res.append("ASII")
    if "LIFE" in res:
        res.remove("LIFE")
    if "NASA" in res:
        res.remove("NASA")
    if "WIKA" in res:
        if re.search(r"\bwika gedung\b", title.lower()):
            res.remove("WIKA")
            res.append("WEGE")
    return res

def find_next_date(date_time, dates):
    next_date = date_time + datetime.timedelta(days = 1)
    while next_date not in dates:
        next_date += datetime.timedelta(days = 1)
    return next_date

def get_tdate(date_time, hour, dates):
    # Check if date is before trading hour
    # Check next period is in business day
    # If in business date use according to the original rule
    # If not use the
    if hour <= 8:
        if date_time in dates:
            return date_time
        else:
            return find_next_date(date_time)
    else:
        return find_next_date(date_time)