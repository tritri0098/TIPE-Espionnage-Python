from radiant_mlhub import Dataset

API_TOKEN = 'ebffd68f3c2982ddd9475075e34d24de0875a86f6499975fb10421e3ba8ad942'

dataset = Dataset.fetch('ref_landcovernet_eu_v1')

dataset.download(output_dir="F:/Documents/Prepa/TIPE/espionnage/IA/datasets/training/LandCoverNet/")
