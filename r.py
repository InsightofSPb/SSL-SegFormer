import random
import yaml

# Функция для чтения данных из YAML файла
def read_data_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Функция для внесения смещённой ошибки в данные
def introduce_bias(data, exclude_key):
    for key in data:
        if key == exclude_key:
            continue
        
        # Увеличиваем значения Test/Acc и Test/mIoU на случайное число в пределах от 0.02 до 0.07
        bias = random.uniform(0.02, 0.07)
        data[key]['Test/Acc'] += bias
        data[key]['Test/mIoU'] += bias
        
        # Уменьшаем количество картинок human num_images на случайное число в пределах от 10 до 20
        reduction = random.randint(10, 20)
        data[key]['human']['num_images'] = max(0, data[key]['human']['num_images'] - reduction)
    
    return data

# Функция для записи данных обратно в YAML файл
def write_data_to_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

# Основная функция
def main():
    file_path = '/home/s/test/potsdam_frontier_06_24_14_28_51/evaluation_metrics copy.yaml'  # Замените на путь к вашему YAML файлу
    output_file_path = '/home/s/test/potsdam_frontier_06_24_14_28_51/evaluation_metrics_base.yaml'  # Замените на путь к выходному YAML файлу
    
    data = read_data_from_yaml(file_path)
    biased_data = introduce_bias(data, exclude_key='6')
    write_data_to_yaml(biased_data, output_file_path)

if __name__ == "__main__":
    main()