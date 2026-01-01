from src.data.dataset import ROPDataset
from src.data.transforms import get_transforms

dataset = ROPDataset(
    csv_file="data/splits/train.csv",
    transforms=get_transforms(train=True)
)

img, label = dataset[0]
print(img.shape, label)
