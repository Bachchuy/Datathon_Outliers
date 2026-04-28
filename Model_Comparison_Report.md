# 📊 Model Comparison & Performance Report — Hà Quốc Khánh

> **Mục tiêu**: So sánh các phương pháp dự báo và chốt phương án nộp bài tốt nhất cho Datathon 2026.

| Phiên bản (Version) | Mô hình (Model) | Kỹ thuật chính | Kết quả Kaggle | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **v10 (Best)** | **Pseudo-Labeled** | **Refinement trên v9** | **Tốt nhất (Top 1)** | ✅ **Chốt nộp** |
| **v9** | Ensemble v9 | Fourier + YoY + Seasonal | Rất tốt | Base model |
| **v12** | Multiplier | Trend Scaling (10-20%) | Thấp hơn v9 | Không hiệu quả |
| **v11** | Breakthrough | COGS Feature (Dummy) | Rất tệ | Thất bại |

## 💡 Kết luận Cuối cùng
- **Mô hình Vô địch**: Kỹ thuật **Pseudo-Labeling (v10)** đã chứng minh được sức mạnh vượt trội. Bằng cách để mô hình tự học lại trên những dự báo tốt nhất của chính nó (v9), chúng ta đã triệt tiêu được các sai số ngẫu nhiên và khớp hoàn hảo với xu hướng của tập Test.
- **Bài học kinh nghiệm**: 
    1. Không nên tin vào `COGS` trong `sample_submission` vì đó là dữ liệu giả (Dummy).
    2. Việc scale thủ công (v12) không hiệu quả bằng việc để mô hình tự học cấu trúc dữ liệu qua Pseudo-labeling. Tỷ suất lợi nhuận và xu hướng đã được LightGBM học một cách tự nhiên và chính xác nhất ở bản v10.

## 📉 Các chỉ số kỹ thuật của bản v10
- **MAE**: Thấp nhất toàn dự án.
- **Tính ổn định**: Cao nhất nhờ sự kết hợp giữa Seasonal Profile và Pseudo-labeling.
- **Tính thực tiễn**: Đảm bảo Revenue >= COGS xuyên suốt 18 tháng.
