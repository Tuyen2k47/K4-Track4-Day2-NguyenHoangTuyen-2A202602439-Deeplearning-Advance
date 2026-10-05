# BÀI LÀM LAB DAY 2 — DEEP LEARNING ADVANCED

* **Học viên:** Nguyễn Hoàng Tuyên
* **Mã số sinh viên (MSSV):** 2A202602439
* **Thư mục bài nộp:** `submissions/2A202602439_nguyen_hoang_tuyen/`

---

## 1. Cấu trúc Thư mục Bài Nộp

Bài làm được tổ chức đúng cấu trúc quy định tại `README.md mục 5`:

```
submissions/2A202602439_nguyen_hoang_tuyen/
├── README.md              # File này: hướng dẫn chạy lại, môi trường, link notebook
├── report.md              # Báo cáo thực nghiệm khoa học 9 mục đầy đủ
├── results.xlsx           # File Excel tổng hợp 7 sheet số liệu
├── curves/                # 16 biểu đồ huấn luyện chi tiết cho từng exp_id (.png)
│   ├── B01_resnet50.png
│   ├── B02_resnext50_32x4d.png
│   ├── B03_convnext_tiny.png
│   ├── B04_deit_small_patch16_224.png
│   ├── B05_mobilenetv3_large_100.png
│   ├── T00_convnext_tiny_baseline.png
│   ├── T01_convnext_tiny_frozen.png
│   ├── T02_convnext_tiny_randaug.png
│   ├── T03_convnext_tiny_cutmix.png
│   ├── T04_convnext_tiny_focal.png
│   ├── T05_convnext_tiny_label_smoothing.png
│   ├── T06_convnext_tiny_ema.png
│   ├── T07_convnext_tiny_combo_best.png
│   ├── F01_seed0_final_convnext_tiny_s0.png
│   ├── F01_seed1_final_convnext_tiny_s1.png
│   └── F01_seed2_final_convnext_tiny_s2.png
├── predictions/           # Dự đoán test & val chính thức (dùng để eval.py chấm lại)
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── F01_uncal_seed0_test.csv
│   ├── F01_uncal_seed1_test.csv
│   ├── F01_uncal_seed2_test.csv
│   ├── F01_seed0_val.csv
│   ├── F01_seed1_val.csv
│   ├── F01_seed2_val.csv
│   ├── T00_seed0_test.csv
│   ├── T00_seed1_test.csv
│   └── T00_seed2_test.csv
└── code/                  # Toàn bộ mã nguồn hoàn chỉnh kế thừa từ starter/
    ├── dataset.py         # Xử lý dữ liệu, kiểm tra S1-S6, DataLoader
    ├── model.py           # Backbone, freeze, 3 nhóm tham số optimizer, params/GMAC
    ├── losses.py          # Label smoothing, Focal loss, CutMix/Mixup
    ├── benchmark.py       # Đo độ trễ chuẩn p50/p95/p99 với synchronize GPU
    ├── inference.py       # TTA, Ensemble, Temperature Scaling, BN fusion
    ├── train.py           # Vòng huấn luyện Config chung, AMP, Cosine warmup, EMA
    ├── test_code.py       # Unit test toàn diện (đạt 8/8 tests)
    └── lab_day2.ipynb     # Notebook thực thi trên Kaggle / Colab
```

---

## 2. Môi trường & Phiên bản Thư viện

* **Python:** `3.10+` (hoặc `3.11`)
* **PyTorch:** `torch >= 2.0.0`
* **Torchvision:** `torchvision >= 0.15.0`
* **Timm:** `timm >= 0.9.2` (dùng cho các backbone `convnext_tiny`, `deit_small`, `resnext50_32x4d`...)
* **Pandas & NumPy:** `pandas >= 2.0.0`, `numpy >= 1.24.0`
* **Scikit-learn:** `scikit-learn >= 1.2.0`
* **OpenPyXL:** `openpyxl >= 3.1.0` (dùng để đọc/ghi `results.xlsx`)
* **Matplotlib:** `matplotlib >= 3.7.0`

Cài đặt nhanh:
```bash
pip install torch torchvision timm pandas numpy scikit-learn openpyxl matplotlib
```

---

## 3. Link Notebook Thực thi (Kaggle & Google Colab)

* **Kaggle Notebook (Khuyên dùng):** [Kaggle Notebook - DeepWeeds Lab Day 2](https://www.kaggle.com/) *(mở file `code/lab_day2.ipynb` trên Kaggle, bật GPU T4 x2 hoặc P100 và Internet)*
* **Google Colab:** [Google Colab - DeepWeeds Lab Day 2](https://colab.research.google.com/) *(chạy với GPU T4, nạp trực tiếp `code/lab_day2.ipynb`)*

---

## 4. Hướng dẫn Thứ tự Chạy Thực nghiệm

### Bước 0: Tải dữ liệu & Chuẩn bị
Tải file `images.zip` từ Zenodo và nhãn Fold 0 từ GitHub:
```bash
# Tạo thư mục data
mkdir -p data/labels
# Tải nhãn
BASE="https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
for f in labels train_subset0 val_subset0 test_subset0; do
    wget -q -O data/labels/$f.csv $BASE/$f.csv
done
# Tải ảnh (MD5: b7b30f96d466fba86016aa5a26606e0f)
wget -O data/images.zip "https://zenodo.org/records/7939060/files/images.zip?download=1"
unzip -q -n data/images.zip -d data/
```

### Bước 1: Chạy Unit Tests để kiểm tra tính đúng đắn của mã nguồn
```bash
cd submissions/2A202602439_nguyen_hoang_tuyen/code
python -m unittest test_code.py -v
cd ../../..
```

### Bước 2: Huấn luyện các thí nghiệm qua CLI hoặc Notebook
Chạy thí nghiệm qua script `train.py`:
```bash
# 1. So sánh Backbone (B01 - B05)
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=B01 backbone=resnet50 seed=0
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=B03 backbone=convnext_tiny seed=0

# 2. Khảo sát công thức huấn luyện (T00 - T07)
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=T03 backbone=convnext_tiny mix=cutmix mix_alpha=1.0 seed=0
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=T07 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=0

# 3. Chung kết 3 hạt giống (F01 seed 0, 1, 2)
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=0 save_test_predictions=True
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=1 save_test_predictions=True
python submissions/2A202602439_nguyen_hoang_tuyen/code/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=2 save_test_predictions=True
```

### Bước 3: Chạy Đánh giá và Chấm điểm tự động qua `eval.py`
```bash
# 1. Tính chỉ số chi tiết cho F01
python eval.py score --pred "submissions/2A202602439_nguyen_hoang_tuyen/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

# 2. Tự chấm điểm phần I của RUBRIC (Mục I1 - I5)
python eval.py grade \
    --final "submissions/2A202602439_nguyen_hoang_tuyen/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602439_nguyen_hoang_tuyen/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602439_nguyen_hoang_tuyen/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submissions/2A202602439_nguyen_hoang_tuyen/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 12.1 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv
```

---

## 5. Danh sách Hạt giống (Seeds) đã Sử dụng
* **Sàng lọc Backbone (Bước 1) & Khảo sát Công thức (Bước 2):** Sử dụng `seed = 0` cố định cho mọi thí nghiệm để đảm bảo tính so sánh công bằng.
* **Chung kết (Bước 4):** Sử dụng **3 hạt giống độc lập** `seed = 0, 1, 2` cho cả cấu hình chung kết ($F01$) và mốc nền ($T00$), báo cáo ở dạng `mean ± std` ($ddof=1$).
