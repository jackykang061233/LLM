from transformers import AutoTokenizer, logging

logging.set_verbosity_error()

def num_tokens_from_string(document: str, tokenizer: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokens = tokenizer.encode(document, add_special_tokens=True)
    
    return len(tokens)

if __name__ == '__main__':
    text = '<s>[INST] \
        請根據以下合約條款與法律條款的對應關係，合併比對結果。如果一條合約條款對應到多條法律條款，請將每個對應關係分開來呈現，但保持在同一個段落中。請使用清晰的分層格式來展示每個對應關係，並提供對應的理由。如果有多段比對結果，請將它們全部整合在一起，確保每條合約條款的所有相關法律條款都被包含。\
        範例:    \
        合約第一條對應法律第二條，理由: 兩者都涉及租金支付的規定。\
        合約第二條對應法律第一條，理由: 兩者都提到乙方需保證設備的正常運作。\
        合約第三條無對應法律，理由: 法規中未提及合約期間內更改設計規格的規定。\
        合約第四條無對應法律，理由: 法規中未提及合資訊服務供應商的資通安全事件通報機制。\
        ------------\
        合約第一條對應法律第三條，理由是兩者都要求遵守國家安全相關法規。\
        合約第二條無對應法律，理由是法規清單中沒有提及設備的正常運作\
        合約第三條無對應法律，因為法規清單中沒有提及合約期間內更改設計規格的相關規定。\
        合約第四條對應法律第四條第四款，理由:兩者都涉及資訊服務供應商的資通安全事件通報機制。\ [/INST]\
        合約第一條對應法律條款如下:\
            - 對應法律第二條。理由：涉及租金支付的規定。\
            - 對應法律第三條。理由：要求遵守國家安全相關法規。\
        合約第二條對應法律條款如下: 對應法律第一條。理由：提到乙方需保證設備的正常運作。\
        合約第三條無對應法律條款。理由：法規中未提及合約期間內更改設計規格的規定。\
        </s>\
        [INST]'
    print(num_tokens_from_string(text, 'MediaTek-Research/Breeze-7B-Instruct-v1_0'))
