from fusion_pipeline.ocr.ocr_detector import detect_ocr_segments
from fusion_pipeline.ocr.ocr_normalise import normalise_ocr_text

raw_segs = detect_ocr_segments('shared/test-videos/IT94xC35u6k.mp4', sample_every_n_seconds=11.1)
print(f'Total raw OCR segments: {len(raw_segs)}')
for s in raw_segs[:40]:
    norm = normalise_ocr_text(s.text)
    print(f'[{s.timestamp:6.1f}s] Conf: {s.confidence:.2f} | Raw: "{s.text}" -> Norm: "{norm}"')
