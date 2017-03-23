import pandas as pd
import datetime

etsng_to_okved = {} #Создаём словарик кодов
code = pd.read_excel('code1.xlsx')
for i in range(code.shape[0]):
    etsng = code.iloc[i,0].strip()[:2]
    okved = code.iloc[i,2].astype(int)
    etsng_to_okved[etsng] = okved
                  
unique_okveds = set(etsng_to_okved.values())
print('уникальные индетификаторы ОКВЭД:', unique_okveds)

columns = ['Вид перевозки', 'Месяц', 
                'Государство отправления', 
                'Государство назначения', 
                'Субъект Федерации отправления', 
                'Субъект Федерации назначения', 
                'Код груза', 'Код груза по ОКВЭД', 
                'Категория отправки', 
                'Признак собственности', 'Признак аренды',
                'Род вагона', 'Код валюты', 
                'Объем перевозок', 'Грузооборот', 
                'Провозная плата', 'Вагоно-км', 
                'Средняя дальность перевозки', 'Тип парка']
    

def result_to_okved_tbl(result_tbl_file, okved_tbl_pattern):
    filewriter_dict = {}
    okved_file_lines = {}
    for okved in unique_okveds:
        # filename = 'okved_' + str(okved) + '_2017.csv'
        filename = okved_tbl_pattern % okved
        # todo заменить excel на csv
        #filewriter_dict[okved] = pd.ExcelWriter(filename, engine='openpyxl')
        filewriter_dict[okved] = open(filename, 'w', encoding='cp1251')
        okved_file_lines[okved] = 0
    # a = pd.read_excel('my_file.xlsx') #Загружая y=нужные столбцы для последующей обработки
    
    # a.to_csv('your_csv.csv', columns=columns, encoding='utf-8')
    start = datetime.datetime.now()
    
    b = pd.read_table(result_tbl_file, sep=';', chunksize = 100000,
                      encoding='cp1251', usecols=columns)
    processed_lines_count = 0
    for chunk in b:
        dic = {} #Записываем все строки с одинаковыми кодами в файл
        #print(chunk.shape)
        for i in range(chunk.shape[0]):
            etsng_full = str(chunk.iloc[i,6])
            etsng = etsng_full.strip()[:2]
            try:
                okved = etsng_to_okved[etsng]
            except KeyError as e:
                print('в строке', i + processed_lines_count, 'найден неизвестный код ЕТСНГ')
                print(chunk.iloc[i,6])
                continue;
            # chunk.iloc[i,9] = okved
            # print(chunk[i,17])            
            #if okved in dic.keys():
            ##    dic[okved] = dic[okved].append(chunk.iloc[i:i+1, :])
            #else:
            #    dic[okved] = chunk.iloc[i:i+1, :]
            data = chunk.iloc[i : i + 1, :]
        # for okved, data in dic.items(): #data - те строки таблицы chunk, которые имеют оквед okved
            k = okved_file_lines[okved] # k - текущее количество строк в файле с рассматриваемым окведом
            header = k == 0 # если было 0 строк значит еще нужно шапку таблицы записать
            #data.to_excel(filewriter_dict[okved], startrow=k, index=False, header=header)
            data.to_csv(filewriter_dict[okved], index=False, header=header, encoding='cp1251')
    
            # записали все строки data + возможно шапку
            okved_file_lines[okved] = k + len(data) + (1 if header is True else 0)
        
        processed_lines_count += len(chunk)
        print('Обработано', processed_lines_count, 'строк')
        
        
    for fw in filewriter_dict.values():
        fw.close()
    
    td = datetime.datetime.now() - start
    print('обработка заняла ', td)

fileslist = [('RESULT%d.csv' % year, 'okved_%%d_%d.csv' % year) for year in range(2012, 2015)]
for inp, out in fileslist:
    print('Обрабатываю файл ', inp)
    result_to_okved_tbl(inp, out)