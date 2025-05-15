from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import os
import sys
import PrimaryStepThree as data
import time
import faiss
from sklearn.metrics.pairwise import cosine_similarity
#поочередно выводит данные: super_vector, min_price, double max_price, регионы работы, ОКПД2(claster), тендеры поставщика 
suplier_info = data.main_start(265917)
path_tendery_test = pd.read_csv("test.csv")

electro_words = [
    "элек", "обор", "кабе", "пров", "тран", "гене", "двиг", "щит",
    "авто", "расп", "конт", "розе", "выкл", "реле", "свет", "осве",
    "ламп", "кабл", "пита", "шну", "разе", "пане", "шкаф", "эле",
    "нагре", "датч", "плат", "блок", "приб", "устр", "акку", "заря",
    "вент", "инве", "стаб", "пере", "резе", "изол", "счет", "исто",
    "фиде", "мото", "пред", "дрос", "разъ", "усил", "выпр", "токо",
    "воль", "сете"
]



def is_OKPD2_same(suplier_info, new_tenders):
	Exsit_OKPD2 = suplier_info[4]
	valid_okpd2_codes = set(Exsit_OKPD2.keys())
	#Находит все совпадающие ОКПД2 коды
	mask = new_tenders["okpd2_code"].isin(valid_okpd2_codes)
	matching_rows = new_tenders[mask]
	return matching_rows

def is_price_avaiable(df_okpd2, min_price,double_max_price):
	# Создаем булеву маску
    mask = (df_okpd2['lot_price'] >= min_price) & (df_okpd2['lot_price'] <= double_max_price)
    # Фильтруем DataFrame с помощью булевой маски
    matching_rows = df_okpd2[mask]
    return matching_rows

def is_region_apply(df_sum, regions):
	mask = df_sum["region_code"].isin(regions)
	matching_rows = df_sum[mask]
	return matching_rows

def find_similar_tenders(df_regions, super_vector, bert_model_name='bert-base-uncased', top_n=10):
    #Находит top_n наиболее похожих тендеров в df_regions на основе сравнения
    #векторизованных названий тендеров с super_vector.

    # 2. Загрузка BERT токенизатора и модели
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertModel.from_pretrained(bert_model_name)

    # 3. Функция для векторизации названия тендера с помощью BERT
    def vectorize_text(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].detach().numpy()

    # 1. Проверка размерности super_vector
    if super_vector.ndim == 1:
        super_vector = super_vector.reshape(1, -1)

    tendr_names_sort = []

    # 2. Векторизация названий тендеров
    tender_names = df_regions['purchase_name'].tolist()


    for z in tender_names:
        if not(isinstance(z,str)):
            z = str(z)
        for i in z.split():
            for j in electro_words:
                if i.lower()[:4] == j.lower():
                    tendr_names_sort.append(i)
                    break


    tender_vectors = np.array([vectorize_text(name) for name in tendr_names_sort])

    # 3. Вычисление косинусного сходства
    similarity_scores = cosine_similarity(tender_vectors, super_vector)


    
    # 4. Получение индексов top_n наиболее похожих тендеров
    top_indices = np.argsort(similarity_scores[:, 0])[::-1][:top_n]


    # 5. Извлечение строк из df_regions
    similar_tenders_df = df_regions.iloc[top_indices].copy()  # Создаем копию, чтобы избежать SettingWithCopyWarning

    # 6. Добавление колонки с косинусным сходством
    similar_tenders_df['similarity_score'] = similarity_scores[top_indices, 0]

    return similar_tenders_df

fstart = time.time()

#Все подходящие по ОКПД2 тендеры
start = time.time()
df_okpd2 = is_OKPD2_same(suplier_info, path_tendery_test)
print(f"Всего тендеров с одинаковым ОКПД2: {len(df_okpd2)}")
end = time.time()
print(f"Выявление одинаковых ОКПД2 кодов заняло: {end-start}")

#Все подходяшие по ОКПД2 и сумме тендеры
start = time.time()
df_sum = is_price_avaiable(df_okpd2, suplier_info[1], suplier_info[2])
print(f"Всего тендеров с подходящей суммой: {len(df_sum)}")
end = time.time()
print(f"Выявление подходящих сумм заняло: {end-start}")

#Все подходящие по ОКПД2, сумме и регионам тендеры 
start = time.time()
df_regions = is_region_apply(df_sum, suplier_info[3])
print(f"Всего тендеров в подходящих регионах: {len(df_regions)}")
end = time.time()
print(f"Выявление подходящих регионов заняло: {end-start}")

#Все схожие названия
df_final = find_similar_tenders(df_regions,suplier_info[0])
print(f"Всего подходящих тендеров: {len(df_final)}")
print(df_final["purchase_name"])
fend = time.time()
print(f"Выявление всех подходящих тендеров заняло: {fend-fstart}")
print(f'Косинусные коэффициенты в процентах:')
print(f'{df_final['similarity_score']*100}')

