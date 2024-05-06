import re
from count_token import num_tokens_from_string
 
class DocumentPreprocess:
    def format_legal_document(self, document_text):
        # Replace Chinese numerals
        formatted_text = re.sub(r'[：:]', '', document_text)
   
        formatted_text = re.sub(r'^第\s?(\d+)\s?條$', lambda x: f'第{self.arabic_to_chinese(x.group(1))}條\n', formatted_text, flags=re.MULTILINE)
        formatted_text = re.sub(r'^第(?:[一二三四五六七八九十百千万]+)條(?:之[一二三四五六七八九十百千万]+)?\s?', lambda x: x.group(0) +'\n', formatted_text, flags=re.MULTILINE)
 
        return formatted_text
 
    def arabic_to_chinese(self, arabic_numeral):
        chinese_numerals= {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
        units= ['', '十', '百', '千']
 
        arabic_numeral = int(arabic_numeral)
        if arabic_numeral < 10:
            return chinese_numerals[arabic_numeral]
        elif arabic_numeral < 20:
            return f'十{chinese_numerals[arabic_numeral%10]}' if arabic_numeral != 10 else f'十'
        elif arabic_numeral < 100:
            ten_digit = arabic_numeral // 10
            unit_digit = arabic_numeral % 10
            if unit_digit == 0:
                return f'{chinese_numerals[ten_digit]}十'
            else:
                return f'{chinese_numerals[ten_digit]}十{chinese_numerals[unit_digit]}'
        else:
            num_str = str(arabic_numeral)
            num_len = len(num_str)
            result = ''
            for i in range(num_len):
                digit = int(num_str[i])
                if digit != 0:
                    result = chinese_numerals[digit] + units[num_len-i-1]
                else:
                    if i < num_len - 1 and int(num_str[i+1]) != 0:
                        result += chinese_numerals[digit]
            return result
 
    def split_legal_document(self, document_text, model, max_length):
        # 使用正則表達式匹配 "第X條"，其中X是一個或多個數字
        articles = re.split(r'(第(?:[一二三四五六七八九十百千万]+)條(?:之[一二三四五六七八九十百千万]+)?\s?)', document_text)
        segments = []
        current_segment = ""
        
        for i in range(1, len(articles), 2):  # 跳過分割的文本，只取 "第X條" 和其後的文本
            article = articles[i] + articles[i+1]  # 將 "第X條" 和其後的文本結合
            article_length = num_tokens_from_string(document=article, tokenizer=model)
           
            # 如果加上這條法律會超出2500字，就先將目前的段落儲存，然後重置段落
            if len(current_segment) + article_length > max_length:
                if current_segment:  # 確保不是空的
                    segments.append(current_segment)
                current_segment = article
            else:
                current_segment += article
 
        # 將最後一個段落加入（如果不是空的）
        if current_segment:
            segments.append(current_segment)
 
        return segments
 
    def split_text_with_overlap(self, text, model, num_chunks, overlap):
        chunk_size = num_tokens_from_string(document=text, tokenizer=model) // num_chunks
        chunks = []
       
        for i in range(0, num_chunks):
            start_index = i*chunk_size
            # Ensure that we don't go beyond the list's length
            end_index = min(start_index + chunk_size + overlap, num_tokens_from_string(document=text, tokenizer=model))
            chunks.append(text[start_index:end_index])
           
            # Adjust the end index for the last chunk to include the remaining text
            if i == num_chunks-1:
                chunks[i] = text[start_index:]
       
        return chunks
   
    def main(self, document, model, max_length):
        document_text = self.format_legal_document(document)
        document_segments = self.split_legal_document(document_text, model, max_length)
 
        return document_segments


# if __name__ == '__main__':
#     with open('document/legal.txt') as f:
#         file = f.read()
#     doc = file.split('----------------')[0]
#     d = DocumentPreprocess()
#     segs = d.main(doc, 'MediaTek-Research/Breeze-7B-Instruct-v1_0', 2000)
    
#     for seg in segs:
#         print(seg)
#         input()