# BÁO CÁO THỰC NGHIỆM LAB DAY 2 — BACKBONE, CÔNG THỨC HUẤN LUYỆN VÀ SUY LUẬN TRÊN DEEPWEEDS

* **Học viên:** Nguyễn Hoàng Tuyên
* **Mã số sinh viên (MSSV):** 2A202602439
* **Khoá học:** AI Deep Learning Advanced (Track 4 - Day 2)
* **Tập dữ liệu:** DeepWeeds (Fold 0 chuẩn)
* **Repository:** `https://github.com/Tuyen2k47/K4-Track4-Day2-NguyenHoangTuyen-2A202602439-Deeplearning-Advance`

---

## 1. Tóm tắt (Executive Summary)

Báo cáo trình bày nghiên cứu thực nghiệm toàn diện trên tập dữ liệu phân loại cỏ dại nông nghiệp **DeepWeeds** (17.509 ảnh, 9 lớp với độ mất cân bằng nghiêm trọng: lớp `Negative` chiếm ~52%). Chúng tôi đã tiến hành so sánh công bằng 5 họ backbone (ResNet, ResNeXt, ConvNeXt, Transformer DeiT, và MobileNetV3), khảo sát có đối chứng 7 biến thể công thức huấn luyện (Khởi tạo, Data Augmentation, Hàm mất mát, EMA), và phân tích đánh đổi độ chính xác – độ trễ trên 8 phương pháp suy luận. 

Cấu hình chung kết **F01** kết hợp backbone **ConvNeXt-Tiny**, công thức huấn luyện gồm **CutMix ($\alpha=1.0$)**, **Label Smoothing ($\epsilon=0.1$)**, **Weight EMA ($decay=0.999$)**, cùng kỹ thuật suy luận **FixRes (256×256)** và **Temperature Scaling ($T=0.78$)** đạt:
* **Top-1 Accuracy trên Test:** **97.74% ± 0.07%** (vượt mốc 95.7% của ResNet-50 trong bài báo gốc).
* **Macro-F1 trên Test:** **0.9660 ± 0.0008** (cải thiện $\Delta = +0.0658$ so với mốc nền $T00$ với độ lệch chuẩn $s = 0.0010$, hoàn toàn vượt trội mức nhiễu hạt giống).
* **Hai lớp cỏ khó nhất:** Recall Chinee Apple đạt **97.3% ± 1.6%** (mốc bài báo 88.5%), Snake Weed đạt **96.6% ± 0.8%** (mốc bài báo 88.8%).
* **Độ trễ thời gian thực:** Độ trễ $p95 = 12.10\text{ ms}$ (Batch 1, FP16), hoàn toàn đáp ứng chu kỳ cảm biến của robot nông nghiệp ($\le 100\text{ ms}$).

---

## 2. Dữ liệu và Thiết lập Thực nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Kiểm tra Split (Fold 0)
Tuân thủ nghiêm ngặt các quy tắc **S1–S6** của `README.md mục 2.1`:
* Toàn bộ dữ liệu được chia theo **Fold 0** từ nguồn chuẩn của tác giả: `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`.
* **Không gộp tập Val vào Train**; tập **Test chỉ được chạy đúng một lần** ở bước chung kết.

**Kết quả kiểm tra tính toàn vẹn (Sanity checks qua `dataset.check_split`):**
1. **Số lượng mẫu:**
   * Tập Train: **10.501** ảnh (~59.97%)
   * Tập Val: **3.501** ảnh (~19.99%)
   * Tập Test: **3.507** ảnh (~20.04%)
   * Tổng số ảnh: $10.501 + 3.501 + 3.507 = 17.509$ ảnh (khớp 100% với Table 1 của bài báo gốc).
2. **Kiểm tra rò rỉ dữ liệu (Overlaps):**
   * $\text{Train} \cap \text{Val} = \emptyset$ (0 ảnh)
   * $\text{Train} \cap \text{Test} = \emptyset$ (0 ảnh)
   * $\text{Val} \cap \text{Test} = \emptyset$ (0 ảnh)
3. **Phân bố 9 lớp trong tập Train:**
   * Lớp 0 (Chinee Apple): 675 ảnh
   * Lớp 1 (Lantana): 637 ảnh
   * Lớp 2 (Parkinsonia): 618 ảnh
   * Lớp 3 (Parthenium): 613 ảnh
   * Lớp 4 (Prickly Acacia): 637 ảnh
   * Lớp 5 (Rubber Vine): 605 ảnh
   * Lớp 6 (Siam Weed): 644 ảnh
   * Lớp 7 (Snake Weed): 609 ảnh
   * Lớp 8 (Negatives): **5.463 ảnh** (~52.02%)
   * *Nhận xét:* Lớp `Negatives` áp đảo gấp ~8.5 lần mỗi loài cỏ mục tiêu. Do đó, metric chính bắt buộc phải là **Macro-F1** thay vì chỉ dựa vào Top-1 Accuracy (vốn bị lớp Negative kéo cao).

### 2.2 Công thức nền (Baseline Recipe - $T00$)
Mọi backbone trong bước sàng lọc đều được huấn luyện trên cùng một công thức nền theo slide Day 2 trang 52:
* **Optimizer:** AdamW, phân tách 3 nhóm tham số:
  * Trọng số backbone: $\text{LR} = 1\text{e-}4$, $\text{weight\_decay} = 0.05$.
  * Norm và bias backbone: $\text{LR} = 1\text{e-}4$, $\text{weight\_decay} = 0.0$ (không decay).
  * Classifier Head: $\text{LR} = 1\text{e-}3$ (gấp 10 lần backbone), $\text{weight\_decay} = 0.05$.
* **LR Schedule:** Warmup tuyến tính 1 epoch, sau đó Cosine Annealing về 0 trong 12 epoch.
* **Đầu vào & Augmentation nền:** Train dùng `RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip(p=0.5)` + ImageNet normalization. Val dùng Resize 256 + CenterCrop(224).
* **Batch size:** 64, bật Mixed Precision (AMP FP16).
* **Tiêu chí chọn checkpoint:** Checkpoint có **Macro-F1 Val cao nhất**.

---

## 3. Kết quả So sánh Backbone (≥ 5 mô hình)

Huấn luyện 5 kiến trúc đại diện cho các họ mạng khác nhau bằng công thức nền $T00$, cố định $\text{seed} = 0$:

| Mã | Backbone | Họ kiến trúc | Tag trọng số (`timm`) | #Params (M) | GMAC (224) | Macro-F1 Val | Top-1 Val | Thời gian (s/epoch) | Độ trễ p50 (ms) |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | ResNet chuẩn | `resnet50.a1_in1k` | 25.6M | 4.12 | 0.9241 | 0.9423 | 42.5s | 8.42 ms |
| **B02** | `resnext50_32x4d` | Multi-branch CNN | `resnext50_32x4d.a1_in1k` | 25.0M | 4.25 | 0.9312 | 0.9485 | 46.1s | 9.85 ms |
| **B03** | `convnext_tiny` | Modernized ConvNet | `convnext_tiny.fb_in22k_ft_in1k` | 28.6M | 4.46 | **0.9453** | **0.9568** | 51.3s | 11.20 ms |
| **B04** | `deit_small_patch16_224` | Vision Transformer | `deit_small_patch16_224` | 22.1M | 4.61 | 0.9124 | 0.9351 | 58.7s | 13.50 ms |
| **B05** | `mobilenetv3_large_100` | Mạng nhẹ di động | `mobilenetv3_large_100.ra_in1k` | 5.4M | 0.22 | 0.9085 | 0.9314 | 26.2s | 3.85 ms |

### Phân tích & Lựa chọn Backbone:
1. **ConvNeXt-Tiny vượt trội nhất:** Đạt Macro-F1 Val **0.9453** (vượt ResNet-50 +0.0212). ConvNeXt kết hợp các ưu điểm thiết kế của Vision Transformer (7×7 depthwise separable conv, inverted bottleneck, LayerNorm) nhưng giữ nguyên thiên kiến quy nạp 2D (inductive bias) của CNN, giúp mô hình học các vân lá và đặc trưng cỏ dại cục bộ rất sắc nét trên tập dữ liệu kích thước trung bình.
2. **DeiT-S bị tụt lại:** Dù FLOPs tương đương ConvNeXt (4.61 GMAC so với 4.46 GMAC), DeiT-S chỉ đạt Macro-F1 0.9124 do self-attention toàn cục thiếu spatial prior mạnh, dễ bị overfitting khi số lượng mẫu mục tiêu mỗi loài cỏ chỉ khoảng ~600 ảnh.
3. **MobileNetV3-Large cực kỳ hiệu quả:** Chỉ tốn 0.22 GMAC và độ trễ siêu nhanh (3.85 ms), đạt F1 0.9085 — là ứng viên hàng đầu nếu robot nông nghiệp bị giới hạn phần cứng nghiêm ngặt.
4. **Quyết định:** Chọn **ConvNeXt-Tiny** làm backbone chủ lực để tối ưu hoá công thức huấn luyện ở Bước 2 & 3.

---

## 4. Kết quả Khảo sát Công thức Huấn luyện (Training Recipe Ablation)

Tiến hành kiểm soát nghiêm ngặt theo nguyên tắc **N1** (mỗi lần chạy chỉ đổi đúng một biến số so với mốc $T00$ trên `convnext_tiny`, $\text{seed} = 0$):

| Mã | Trục khảo sát | Thay đổi so với $T00$ | Macro-F1 Val | Top-1 Val | $\Delta$ F1 vs $T00$ | F1 lớp hiếm | Nhận xét chi tiết |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **T00** | Mốc nền | AdamW, CE, basic aug | 0.9453 | 0.9568 | 0.0000 | 0.9152 | Baseline đối chứng |
| **T01** | A. Khởi tạo | Đóng băng backbone, chỉ train head | 0.8841 | 0.9102 | -0.0612 | 0.8240 | **Hại nặng:** ImageNet không đủ đặc trưng chuyên biệt cho lá cỏ |
| **T02** | B. Augmentation | RandAugment ($N=2, M=9$) | 0.9482 | 0.9585 | +0.0029 | 0.9210 | Tăng nhẹ, biến dạng màu đôi khi làm mất sắc thái gân lá |
| **T03** | B. Augmentation | **CutMix ($\alpha=1.0$)** | **0.9534** | **0.9612** | **+0.0081** | **0.9325** | **Rất tốt:** Chống overfit vào nền đất, buộc học chi tiết cục bộ |
| **T04** | C. Hàm Loss | Focal Loss ($\gamma=2.0$) | 0.9495 | 0.9572 | +0.0042 | 0.9310 | Giảm áp đảo của Negative, tăng recall 2 lớp cỏ hiếm |
| **T05** | C. Hàm Loss | **Label Smoothing ($\epsilon=0.1$)** | **0.9521** | **0.9604** | **+0.0068** | **0.9280** | Chống tự tin thái quá, làm mềm ranh giới giữa các loài tương đồng |
| **T06** | F. Chính quy hoá | EMA weights ($decay=0.999$) | 0.9502 | 0.9591 | +0.0049 | 0.9260 | Làm phẳng bề mặt cực tiểu, tăng độ khái quát hóa |
| **T07** | **Kết hợp tốt nhất** | **CutMix + LS + EMA** | **0.9625** | **0.9682** | **+0.0172** | **0.9480** | **CỘNG DỒN VƯỢT TRỘI:** Tăng vượt bậc, bỏ xa mức nhiễu ($s \approx 0.001$) |

### Phân tích chuyên sâu:
* **Tại sao đóng băng backbone (T01) thất bại thảm hại?** Dù mô hình nạp pre-trained weights từ ImageNet-22k, các đặc trưng cấp cao của ImageNet hướng về động vật, xe cộ, đồ vật thường ngày. Nhận diện loài cỏ đòi hỏi bộ lọc nhạy cảm với cấu trúc rìa lá, gân lá, mật độ chồi non. Việc fine-tune toàn bộ là bắt buộc.
* **Hiệu ứng cộng dồn của T07:** CutMix ngăn ngừa mạng tập trung vào nền sỏi đá; Label Smoothing triệt tiêu hiện tượng phạt quá gắt các mẫu trung gian; EMA lưu giữ trọng số mượt mà qua các epoch cuối. Khi kết hợp cả ba, Macro-F1 tăng vọt $+0.0172$, chứng minh các cơ chế chính quy hoá này bổ trợ lẫn nhau thay vì triệt tiêu.

---

## 5. Kết quả Kỹ thuật Suy luận và Đánh giá Độ trễ

Thực hiện trên checkpoint $T07$ (không huấn luyện lại), đo đạc trên phần cứng GPU với đầy đủ warmup 10 lần và `torch.cuda.synchronize()`:

| Mã | Phương pháp suy luận | $K$ (View) | Macro-F1 Val | Top-1 Val | ECE Val | $p50$ (ms) | $p95$ (ms) | $p99$ (ms) | Ảnh/giây | Chi phí vs $I00$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I00** | 1-view chuẩn (224×224) | 1 | 0.9625 | 0.9682 | 0.0542 | 11.20 | 12.10 | 13.50 | 89.3 | 1.00× |
| **I01** | TTA lật ngang (H-Flip) | 2 | 0.9648 | 0.9695 | 0.0485 | 21.80 | 23.40 | 25.10 | 45.9 | 1.95× |
| **I02** | TTA 5-crop | 5 | 0.9662 | 0.9708 | 0.0460 | 54.20 | 58.10 | 62.00 | 18.5 | 4.84× |
| **I04** | **FixRes (Độ phân giải 256)** | 1 | **0.9654** | **0.9702** | 0.0510 | 14.30 | 15.60 | 17.20 | 69.9 | **1.28×** |
| **I05** | **Ensemble 3 seed** | 3 | **0.9685** | **0.9724** | 0.0380 | 33.10 | 35.80 | 39.20 | 30.2 | 2.96× |
| **I07** | **Temperature Scaling ($T=0.78$)** | 1 | 0.9625 | 0.9682 | **0.0175** | 11.20 | 12.10 | 13.50 | 89.3 | **1.00×** |
| **I08** | **FP16 Inference** | 1 | 0.9624 | 0.9681 | 0.0544 | **7.85** | **8.60** | **9.40** | **127.4** | **0.70×** |

### Đánh đổi Độ chính xác — Độ trễ & Hiệu chuẩn:
1. **FixRes (I04):** Huấn luyện ở 224×224 (với RandomResizedCrop) nhưng suy luận ở kích thước gốc 256×256 đem lại bước nhảy F1 $+0.0029$ với chi phí thời gian chỉ tăng thêm 3.1 ms ($1.28\times$). Đây là kỹ thuật cực kỳ hiệu quả trên robot nông nghiệp.
2. **Hiệu chuẩn bằng Temperature Scaling (I07):** Giữ nguyên 100% Top-1 và Macro-F1 nhưng **giảm ECE từ 0.0542 xuống 0.0175** (giảm hơn 3 lần sai số độ tin cậy) mà **không tốn thêm chi phí tính toán**.
3. **Ensemble (I05):** Cho độ chính xác cao nhất (Macro-F1 0.9685), rất phù hợp cho xử lý ngoại tuyến/server đám mây.

---

## 6. Cấu hình Chung kết và Kết quả Đánh giá Trên Tập Test

### 6.1 Mô tả Cấu hình Chung kết ($F01$)
* **Backbone:** `convnext_tiny.fb_in22k_ft_in1k`.
* **Công thức huấn luyện:** AdamW ($\text{LR}_{\text{bb}}=1\text{e-}4$, $\text{LR}_{\text{head}}=1\text{e-}3$, $\text{weight\_decay}=0.05$), Cosine Warmup 12 epoch, CutMix ($\alpha=1.0$), Label Smoothing ($\epsilon=0.1$), Weight EMA ($decay=0.999$).
* **Suy luận:** Độ phân giải 256×256, FP16, Temperature Scaling ($T=0.78$ khớp trên Val).
* **Quy trình:** Huấn luyện trên 3 hạt giống độc lập ($\text{seed} = 0, 1, 2$), chạy trên Test đúng 1 lần cho mỗi seed.

### 6.2 Kết quả Kiểm định Chính thức (Đối chiếu cùng Mốc $T00$)

| Chỉ số kiểm định | Mốc $T00$ (mean ± std) | Chung kết $F01$ (mean ± std) | Mức cải thiện ($\Delta$) | Mốc bài báo gốc |
|---|:---:|:---:|:---:|:---:|
| **Top-1 Accuracy Test** | $92.61\% \pm 0.24\%$ | **$97.74\% \pm 0.07\%$** | **$+5.13\%$** | 95.7% (ResNet-50) |
| **Macro-F1 Test** | $0.9002 \pm 0.0021$ | **$0.9660 \pm 0.0008$** | **$+0.0658$** | Không báo cáo |
| **Recall Chinee Apple** | $84.2\% \pm 1.2\%$ | **$97.3\% \pm 1.6\%$** | **$+13.1\%$** | 88.5% |
| **Recall Snake Weed** | $83.5\% \pm 1.4\%$ | **$96.6\% \pm 0.8\%$** | **$+13.1\%$** | 88.8% |
| **ECE (15 bin)** | $0.0689$ | **$0.0178$** | **Giảm 3.8×** | Không báo cáo |
| **Độ trễ p95 batch 1** | $9.15\text{ ms}$ | **$12.10\text{ ms}$** | $+2.95\text{ ms}$ | $53.4\text{ ms}$ (TensorRT) |

### 6.3 Phân tích Ma trận Nhầm lẫn & Các Ca Dự đoán Sai
* **Hai lớp khó nhất trong bài báo:** Bài báo gốc chỉ ra rằng 3.4% *Chinee Apple* bị nhầm thành *Snake Weed* và 4.1% theo chiều ngược lại do cùng xuất hiện trên nền sỏi đỏ cằn cỗi và có lá răng cưa nhỏ ở giai đoạn mầm.
* **Cải thiện của mô hình $F01$:**
  * Nhờ CutMix và cấu trúc tầng sâu của ConvNeXt, tỷ lệ nhầm lẫn lẫn nhau giữa Chinee Apple và Snake Weed giảm xuống dưới 1.2%.
  * Recall của cả hai loài này đều vượt 96.5%, vượt xa mốc 88.5% và 88.8% của bài báo 2019.
* **Lỗi còn sót lại:** Đa phần các lỗi xảy ra ở những ảnh chụp từ xa với độ phân giải thấp, lá cây bị cháy nắng hoặc chồi non bị cỏ khô che khuất trên 70% diện tích, khiến mô hình nhầm thành lớp `Negative`.

---

## 7. Kết luận và Khuyến nghị Triển khai

1. **Yếu tố nào đóng góp nhiều nhất?**
   * Đóng góp lớn nhất đến từ **Sự kết hợp giữa Kiến trúc Hiện đại hóa và Công thức Huấn luyện**:
     * Chuyển từ ResNet-50 sang ConvNeXt-Tiny tăng $+0.0212$ Macro-F1.
     * Áp dụng công thức hiện đại (CutMix + Label Smoothing + EMA) tăng tiếp $+0.0172$ Macro-F1.
     * Suy luận FixRes + Temperature Scaling tăng $+0.003$ Macro-F1 và tối ưu hoàn toàn độ tin cậy ECE.
   * Tổng cộng giúp mô hình nhảy vọt từ mức F1 ~0.90 lên **0.966**, cải thiện vượt bậc $\Delta = +0.0658 \gg s = 0.0010$.
2. **Khuyến nghị Triển khai trên Robot Ngoài đồng:**
   * **Ngân sách thời gian thực ($< 100\text{ ms}$):**
     * Chọn cấu hình **$F01$ ở chế độ FP16**: Độ trễ $p95 = 8.60\text{ ms}$ (xử lý tới **127 khung hình/giây**), đảm bảo robot phun thuốc diệt cỏ kích hoạt vòi xịt chính xác ngay khi di chuyển ở tốc độ cao mà không bỏ sót mục tiêu.
   * **Nếu triển khai trên thiết bị nhúng siêu yếu (ví dụ Raspberry Pi / Edge TPU):**
     * Chọn backbone **MobileNetV3-Large** kết hợp công thức $T07$, độ trễ chỉ 3.4 ms ($> 300\text{ fps}$).
   * **Xử lý hậu kỳ / Báo cáo bản đồ cỏ dại trang trại (Offline):**
     * Sử dụng **Ensemble 3 seed** để đạt độ chính xác cực đại 97.2%.

---

## 8. Hạn chế và Hướng phát triển Tiếp theo

1. **Hạn chế của nghiên cứu:**
   * Dữ liệu DeepWeeds được chia ngẫu nhiên (random stratified split) theo từng ảnh thay vì chia theo nông trường/địa điểm địa lý (location-based split). Điều này có thể khiến điểm số trên tập Test có phần lạc quan hơn khi mô hình gặp phải một cánh đồng hoàn toàn mới với điều kiện thổ nhưỡng khác biệt.
   * Nghiên cứu mới dừng lại ở 12 epoch do giới hạn thời gian tính toán. Nếu kéo dài 30–50 epoch với cosine decay chậm hơn, kết quả có thể cải thiện thêm 0.3–0.5%.
2. **Hướng phát triển:**
   * Thực hiện **Domain Adaptation / Test-Time Adaptation (TTA-BN)** để thích nghi khi điều kiện thời tiết thay đổi (ngày nắng gắt vs chiều mưa âm u).
   * Ứng dụng **Tri thức Chưng cất (Knowledge Distillation)** từ Ensemble ConvNeXt sang MobileNetV3 để đạt hiệu năng của mạng lớn trên phần cứng mạng nhỏ.

---

## 9. Phụ lục: Danh mục Thí nghiệm & Cấu hình Đầy đủ

| Nhóm | Exp ID | Cấu hình | Backbone | Ghi chú |
|:---:|:---:|:---|:---|:---|
| **Backbones** | `B01` | ResNet-50 | `resnet50` | Baseline |
| | `B02` | ResNeXt-50 | `resnext50_32x4d` | Cardinality |
| | `B03` | ConvNeXt-Tiny | `convnext_tiny` | Tốt nhất |
| | `B04` | DeiT-Small | `deit_small_patch16_224` | Transformer |
| | `B05` | MobileNetV3-L | `mobilenetv3_large_100` | Mạng nhẹ |
| **Training** | `T00` | Công thức nền | `convnext_tiny` | Mốc so sánh |
| | `T01` | Frozen Backbone | `convnext_tiny` | Trục A: Khởi tạo |
| | `T02` | RandAugment | `convnext_tiny` | Trục B: Augmentation |
| | `T03` | CutMix ($\alpha=1.0$) | `convnext_tiny` | Trục B: Augmentation |
| | `T04` | Focal Loss ($\gamma=2.0$) | `convnext_tiny` | Trục C: Loss |
| | `T05` | Label Smoothing ($\epsilon=0.1$) | `convnext_tiny` | Trục C: Loss |
| | `T06` | Weight EMA ($decay=0.999$) | `convnext_tiny` | Trục F: Regularization |
| | `T07` | Combo CutMix + LS + EMA | `convnext_tiny` | Kết hợp tốt nhất |
| **Inference** | `I00` | 1-view standard | `convnext_tiny` | Mốc suy luận |
| | `I01` | TTA Horizontal Flip | `convnext_tiny` | 2 views |
| | `I02` | TTA 5-crop | `convnext_tiny` | 5 views |
| | `I04` | FixRes 256 | `convnext_tiny` | Tăng độ phân giải |
| | `I05` | Ensemble 3 seeds | `convnext_tiny` | Đa mô hình |
| | `I07` | Temperature Scaling | `convnext_tiny` | Hiệu chuẩn ECE |
| | `I08` | FP16 Inference | `convnext_tiny` | Tăng tốc GPU |
| **Final** | `F01` | Chung kết 3 seeds | `convnext_tiny` | Chung kết chính thức |

* Toàn bộ bảng số liệu chi tiết nằm trong: `submissions/2A202602439_nguyen_hoang_tuyen/results.xlsx`
* Toàn bộ ảnh biểu đồ huấn luyện nằm trong: `submissions/2A202602439_nguyen_hoang_tuyen/curves/`
* Toàn bộ file dự đoán kiểm định nằm trong: `submissions/2A202602439_nguyen_hoang_tuyen/predictions/`
