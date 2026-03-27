# Hard Rule Pack (Auchan Charte)

File chính: `hard_rules_auchan_vclients.json`
Schema: `hard_rules.schema.json`

## Ghi chú
- Đây là bộ hard-rule đầu tiên (draft) lấy từ các phần đo/kiểm tra được trong charte.
- Ưu tiên dùng cho các check deterministic: màu, typo, kích thước, khoảng cách, hiệu ứng cấm.
- Các rule mềm (mang tính cảm quan/visual) cần để riêng trong `../soft_rule`.

## Khuyến nghị triển khai check
1. PDF parser đọc bbox + màu + font.
2. Rule engine chạy theo `rule_id`.
3. Trả kết quả `pass/fail/partial` + evidence (page, bbox, sampled color/font).
4. Lưu report JSON và PDF annotate.
