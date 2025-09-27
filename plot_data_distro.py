import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm phân chia dữ liệu cho các client theo non-IID Dirichlet
def non_iid_partition_dirichlet(dataset, arr_number_of_client, partition="hetero-dir", alpha=0.1):
    num_clients = len(arr_number_of_client)
    y_train = np.array(dataset.targets)
    N = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(N)
        batch_idxs = np.array_split(idxs, num_clients)
        clients_data = {arr_number_of_client[i]: batch_idxs[i].tolist() for i in range(num_clients)}
        proportions = [1.0 / num_clients] * num_clients
    elif partition == "hetero-dir":
        K = len(np.unique(y_train))
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        clients_data = {arr_number_of_client[i]: idx_batch[i] for i in range(num_clients)}
        proportions = [len(client_data) / N for client_data in idx_batch]

    return clients_data, proportions

# Hàm tải dataset MNIST
def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_data, test_data

# Hàm tải dataset CIFAR-10
def get_cifar10_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    return train_data, test_data

# Hàm tải dataset FashionMNIST
def get_fashion_mnist_data():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    
    return train_data, test_data

# Số lượng client, ví dụ chọn 50 client
num_clients = 50
clients_list = list(range(num_clients))

# Phân chia dữ liệu cho từng dataset
train_data_mnist, _ = get_mnist_data()
clients_data_mnist, _ = non_iid_partition_dirichlet(train_data_mnist, clients_list, partition="hetero-dir", alpha=0.05)

train_data_fashion, _ = get_fashion_mnist_data()
clients_data_fashion, _ = non_iid_partition_dirichlet(train_data_fashion, clients_list, partition="hetero-dir", alpha=0.3)

train_data_cifar, _ = get_cifar10_data()
clients_data_cifar, _ = non_iid_partition_dirichlet(train_data_cifar, clients_list, partition="hetero-dir", alpha=0.5)

# Tính số lượng sample của mỗi client cho từng dataset
client_ids = clients_list
sample_counts_mnist = [len(clients_data_mnist[c]) for c in client_ids]
sample_counts_fashion = [len(clients_data_fashion[c]) for c in client_ids]
sample_counts_cifar = [len(clients_data_cifar[c]) for c in client_ids]

bar_width = 0.75  # Tăng độ rộng cột
gap = 0.3        # Khoảng trống giữa các nhóm

# Tạo mảng vị trí x, mỗi client là 1 nhóm, mỗi nhóm cách nhau (3 * bar_width + gap)
x = np.arange(num_clients) * (3 * bar_width + gap)

plt.figure(figsize=(30, 8))
plt.bar(x - bar_width, sample_counts_mnist, bar_width, label="MNIST", color='skyblue')
plt.bar(x, sample_counts_fashion, bar_width, label="FashionMNIST", color='salmon')
plt.bar(x + bar_width, sample_counts_cifar, bar_width, label="CIFAR-10", color='lightgreen')

plt.xlabel("Client ID", fontsize=30)
plt.ylabel("Number of sample", fontsize=30)
plt.title("Number of sample for each client on each dataset", fontsize=30)
# Hiển thị nhãn mỗi 10 client:
plt.xticks(x[::5], np.array(client_ids)[::5], fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=15)
plt.show()


# Tính số lượng sample của mỗi client cho từng dataset (đã tính ở code trước)
# sample_counts_mnist, sample_counts_fashion, sample_counts_cifar đã có

# MNIST
min_samples_mnist = min(sample_counts_mnist)
max_samples_mnist = max(sample_counts_mnist)
client_min_mnist = sample_counts_mnist.index(min_samples_mnist)
client_max_mnist = sample_counts_mnist.index(max_samples_mnist)
print("MNIST:")
print("  - Client có ít sample nhất: Client", client_min_mnist, "với", min_samples_mnist, "sample")
print("  - Client có nhiều sample nhất: Client", client_max_mnist, "với", max_samples_mnist, "sample\n")

# FashionMNIST
min_samples_fashion = min(sample_counts_fashion)
max_samples_fashion = max(sample_counts_fashion)
client_min_fashion = sample_counts_fashion.index(min_samples_fashion)
client_max_fashion = sample_counts_fashion.index(max_samples_fashion)
print("FashionMNIST:")
print("  - Client có ít sample nhất: Client", client_min_fashion, "với", min_samples_fashion, "sample")
print("  - Client có nhiều sample nhất: Client", client_max_fashion, "với", max_samples_fashion, "sample\n")

# CIFAR-10
min_samples_cifar = min(sample_counts_cifar)
max_samples_cifar = max(sample_counts_cifar)
client_min_cifar = sample_counts_cifar.index(min_samples_cifar)
client_max_cifar = sample_counts_cifar.index(max_samples_cifar)
print("CIFAR-10:")
print("  - Client có ít sample nhất: Client", client_min_cifar, "với", min_samples_cifar, "sample")
print("  - Client có nhiều sample nhất: Client", client_max_cifar, "với", max_samples_cifar, "sample")

