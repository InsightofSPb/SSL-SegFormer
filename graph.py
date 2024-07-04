import matplotlib.pyplot as plt
import yaml

# Функция для чтения данных из YAML файла
def read_data_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Функция для извлечения необходимых данных
def extract_data(data):
    num_images = []
    test_acc = []
    test_miou = []
    
    for key in data:
        num_images.append(data[key]['human']['num_images'])
        test_acc.append(data[key]['Test/Acc'])
        test_miou.append(data[key]['Test/mIoU'])
    
    return num_images, test_acc, test_miou

# Функция для построения графиков
def plot_graphs(num_images1, test_acc1, test_miou1, num_images2, test_acc2, test_miou2, num_images3, test_acc3, test_miou3):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # График Test/Acc от количества human num_images
    axs[0].plot(num_images1, test_acc1, marker='o', linestyle='-', color='b', label='SegFormer-MiT-B1 + SSL')
    axs[0].plot(num_images2, test_acc2, marker='o', linestyle='-', color='g', label='ERFNet + SSL')
    axs[0].plot(num_images3, test_acc3, marker='o', linestyle='-', color='r', label='SegFormer-MiT-B1')
    axs[0].set_title('Accuracy от количества изображений')
    axs[0].set_xlabel('Количество изображений')
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)
    axs[0].legend()

    # График Test/mIoU от количества human num_images
    axs[1].plot(num_images1, test_miou1, marker='o', linestyle='-', color='b', label='SegFormer-MiT-B1 + SSL')
    axs[1].plot(num_images2, test_miou2, marker='o', linestyle='-', color='g', label='ERFNet + SSL')
    axs[1].plot(num_images3, test_miou3, marker='o', linestyle='-', color='r', label='SegFormer-MiT-B1')
    axs[1].set_title('mIoU от количества изображений')
    axs[1].set_xlabel('Количество изображений')
    axs[1].set_ylabel('mIoU')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('comparison_graphs.jpg')  # Сохранение графиков в файл
    plt.show()

# Основная функция
def main():
    file_path1 = '/home/s/test/potsdam_frontier_06_24_14_28_51/evaluation_metrics.yaml'  # Замените на путь к вашему первому YAML файлу
    file_path2 = '/home/s/test/potsdam_frontier_06_24_14_28_51/evaluation_metrics_base.yaml'  # Замените на путь к вашему второму YAML файлу
    file_path3 = '/home/s/test/potsdam_frontier_06_25_17_26_11/evaluation_metrics.yaml'  # Замените на путь к вашему третьему YAML файлу
    
    data1 = read_data_from_yaml(file_path1)
    data2 = read_data_from_yaml(file_path2)
    data3 = read_data_from_yaml(file_path3)
    
    num_images1, test_acc1, test_miou1 = extract_data(data1)
    num_images2, test_acc2, test_miou2 = extract_data(data2)
    num_images3, test_acc3, test_miou3 = extract_data(data3)
    
    plot_graphs(num_images1, test_acc1, test_miou1, num_images2, test_acc2, test_miou2, num_images3, test_acc3, test_miou3)

if __name__ == "__main__":
    main()