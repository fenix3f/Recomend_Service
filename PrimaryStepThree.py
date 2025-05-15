from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import os

def main_start(supplier_id):
    # Пути к файлам
    path_zayavki = "ml_model_participants_post_num.csv"
    path_tendery = "train.csv"


    # 1. Загрузка данных
    df_zayavki = pd.read_csv(path_zayavki, usecols=[0, 1, 2, 3])  
    df_tendery = pd.read_csv(path_tendery, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  

    # 2. Получаем номера тендеров, на которые подавался поставщик
    supplier_tenders = df_zayavki[df_zayavki.iloc[:, 1] == supplier_id].iloc[:, 0].unique()
    print(f"Найдено {len(supplier_tenders)} тендеров, в которых участвовал поставщик. Эта информация неточна, т.к данные разделены 70/30.")

    # 3. Получаем ОКПД2 для этих тендеров
    df_filtered_tenders = df_tendery[df_tendery.iloc[:, 1].isin(supplier_tenders)].dropna(subset=[df_tendery.columns[9]])  # Убираем тендеры без ОКПД2
    print(f"После фильтрации по ОКПД2 осталось {len(df_filtered_tenders)} тендеров.")

    # 4. Группируем тендеры по ОКПД2 (создаем кластеры)
    clusters = {}
    for _, row in df_filtered_tenders.iterrows():
        okpd2 = row.iloc[9]  # ОКПД2 код
        if okpd2 not in clusters:
            clusters[okpd2] = []
        clusters[okpd2].append(row.to_dict())

    # 5. Выводим все уникальные ОКПД2, в которых поставщик участвовал
    print("Все ОКПД2, в которых поставщик участвовал:")
    print(list(clusters.keys()))

    # 6. Загружаем модель и токенизатор для получения эмбеддингов
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("Модели TinyBERT загружены!")

    # 7. Функция для получения вектора для названия тендера
    def get_tinybert_vector(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].detach().numpy()

    # 8. Функция для получения супервектора для каждого ОКПД2
    def get_super_vector_for_okpd2(cluster):
        cluster_df = pd.DataFrame(cluster)  # Преобразуем список в DataFrame
        tender_names = cluster_df.iloc[:, 4].tolist()  # Извлекаем названия тендеров
        tender_vectors = np.array([get_tinybert_vector(name) for name in tender_names])  # Векторизуем все названия
        super_vector = np.mean(tender_vectors, axis=0)  # Средний вектор
        return super_vector

    # 9. Вычисление супервекторов для всех кластеров
    super_vectors = {}
    for okpd2, tenders in clusters.items():
        super_vectors[okpd2] = get_super_vector_for_okpd2(tenders)

    # 10. Выводим супервекторы для каждого ОКПД2
    for okpd2, super_vector in super_vectors.items():
        print(f"Супервектор для ОКПД2 {okpd2} создан")
        print("-" * 50)

    # 11. Выводим информацию о тендерах поставщика
    df_supplier_tenders = df_tendery[df_tendery['pn_lot'].isin(supplier_tenders)]  # Фильтруем тендеры
    # print(f"Найдено {len(df_supplier_tenders)} тендеров для поставщика {supplier_id}")

    # 12. Вычисление максимальной цены тендера и умножение на 2
    max_lot_price = df_supplier_tenders['lot_price'].max()
    min_lot_price = df_supplier_tenders['lot_price'].min()
    max_lot_price_doubled = max_lot_price * 2
    # print(f"Максимальная цена тендера для поставщика: {max_lot_price}")
    # print(f"Максимальная цена тендера, умноженная на 2: {max_lot_price_doubled}")

    # 13. Составление списка уникальных регионов для поставщика
    unique_regions = df_supplier_tenders['region_code'].unique()
    # print(f"Уникальные коды регионов для поставщика: {unique_regions}")

    return super_vector, min_lot_price, max_lot_price_doubled, unique_regions, clusters, supplier_tenders


