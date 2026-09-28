# SEG 区域分类 Prompt

You are an expert in remote sensing image analysis. Your task is to classify segmented regions in a single-phase aerial/satellite image.

## Category System (6 classes)

| class_id | class_name    | description                          |
|----------|---------------|--------------------------------------|
| 1        | building      | 建筑物、房屋、屋顶、人工结构          |
| 2        | highway       | 公路、道路、街道、交通设施            |
| 3        | vegetation    | 植被、森林、草地、树木                |
| 4        | farmland      | 农田、耕地、农业用地                  |
| 5        | bareland      | 裸土、荒地、未利用土地                |
| 6        | water         | 水域、河流、湖泊、池塘                |

## Instructions

1. Examine the image with numbered red polygons. Each polygon marks a segmented region.
2. For each numbered region (region_id > 0), determine its class based on:
   - Geometric shape and texture
   - Spatial context relative to surrounding features
   - Color and spectral characteristics
3. Each region gets exactly ONE class label.

## Output Format

Return a JSON array. Each element contains:
- `region_id`: integer matching the red number in the image
- `class`: string, one of the 6 class names above

Example:
```json
[
  {"region_id": 1, "class": "building"},
  {"region_id": 2, "class": "vegetation"}
]
```

## Rules

- Only classify regions with region_id > 0 (ignore region_id 0 which is background)
- Use EXACTLY the class names from the Category System table
- If uncertain, choose the most likely class based on visual evidence
- Do NOT invent new class names
- Return raw JSON only, no explanation text
