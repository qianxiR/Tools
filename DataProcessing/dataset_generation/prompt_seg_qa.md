# SEG QA 生成 Prompt — Urban Scene Understanding v2

You are generating a visual question-answering dataset for remote sensing segmentation images, with a focus on **urban scene understanding**.

You will receive:
1. An image with numbered red polygons marking segmented regions
2. A structured JSON with region facts: region_id, location, class
3. `num_regions`: total number of valid regions
4. `class_counts`: exact count per class (e.g., {"building": 49, "vegetation": 4})

IMPORTANT: For any counting question, you MUST use `class_counts` as the authoritative source.
Do NOT count regions yourself from the list — use the pre-computed `class_counts` values directly.
The `class_counts` values are ground truth derived from the segmentation mask.

## Urban Scene Understanding Dimensions

When analyzing the scene, consider these 7 dimensions:

### 1. Built Environment (建成环境)
- Building density: high/medium/low
- Building scale: large/medium/small
- Building layout: grid/organic/scattered/clustered
- Example: "The area is dominated by high-density built areas with buildings arranged in a regular grid pattern."

### 2. Transportation Network (交通网络)
- Main roads, highways, railways, intersections
- Road network density and connectivity
- Example: "The road network is well-developed with multiple main roads crossing the area."

### 3. Green Space & Ecology (绿地生态)
- Parks, forests, urban greenery
- Vegetation coverage and distribution
- Example: "Large green areas exist within the region, contrasting with surrounding buildings."

### 4. Water Environment (水体环境)
- Rivers, lakes, reservoirs, canals
- Water body impact on spatial layout
- Example: "A river runs through the study area, significantly influencing the spatial layout."

### 5. Industry & Commerce (工业与商业)
- Industrial parks, warehouses, commercial centers
- Large-scale structures with regular layout
- Example: "The eastern area contains large-scale industrial buildings with regular layout."

### 6. Residential Distribution (居住区分布)
- High-rise residential, villa areas, urban villages
- Settlement patterns and density
- Example: "Multiple residential clusters are distributed in groups with high building density."

### 7. Urban-Rural Pattern (城乡格局)
- Urban core, urban fringe, suburbs, rural settlements
- Transition zone characteristics
- Example: "Built areas and farmland are interspersed, showing typical urban-rural transition zone features."

## Unified Urban Scene Classification

Use these scene type labels for classification questions:
- `high_density_urban`: 高密度城区
- `low_density_urban`: 低密度城区
- `residential_area`: 居住区
- `commercial_area`: 商业区
- `industrial_area`: 工业区
- `transportation_area`: 交通设施区
- `green_space_area`: 绿地区
- `water_area`: 水域区
- `urban_rural_transition`: 城乡结合区
- `mixed_functional_area`: 综合功能区

## Task

Generate a JSON object with a `qa` array containing at least 10 questions covering these types:

### 1. QA (Yes/No) — at least 3

- **CN**: "Is there any visible object in this image?" → answer: "yes" (always, since regions exist)
- **CtW**: "Is there a {class} in this image?" → answer based on whether any region has that class
- **CfW**: "Is the object in the {location} a {class}?" → answer based on region facts
- **Scene**: "Does this image show a high-density urban area?" → answer based on scene type

Rules:
- answer_format: "yes_no"
- answer must be exactly "yes" or "no"

### 2. MCQ (Multiple Choice) — at least 4

- **location**: "Where is the {class} located?" → options with 4 directions, one correct
- **class**: "What type of object is located in the {location}?" → options with 4 classes, one correct
- **count**: "How many {class} regions are visible?" → options with 4 numbers, MUST use `class_counts` as the correct answer
- **scene_type** (at least 1): "What type of urban scene is shown in this image?" → options with 4 scene types

Rules:
- answer_format: "mcq"
- Exactly 4 options (A/B/C/D)
- answer must be exactly one of A/B/C/D

### 3. Caption (Descriptive) — at least 3

- **local**: "Describe the {class} in the {location} of the image."
- **scene_understanding** (at least 1): "Describe the urban scene characteristics of this area."
  - Answer should cover: building density, layout pattern, dominant land use
  - Example: "This area is a typical mixed-function urban district, composed of high-density residential buildings, commercial facilities, and a well-developed road network."
- **overall**: "Provide a comprehensive description of the entire image and all visible objects."

Rules:
- answer_format: "text"
- answer must be a non-empty descriptive text
- Scene understanding captions should use the 7 dimensions above
- Overall captions should identify the urban scene type and key features

## JSON Format

```json
{
  "qa": [
    {
      "id": "q001",
      "type": "qa",
      "question": "Is there a building in this image?",
      "answer": "yes",
      "answer_format": "yes_no"
    },
    {
      "id": "q002",
      "type": "mcq",
      "question": "Where is the building located?",
      "options": {"A": "North", "B": "South", "C": "East", "D": "West"},
      "answer": "A",
      "answer_format": "mcq"
    },
    {
      "id": "q003",
      "type": "mcq",
      "question": "What type of urban scene is shown in this image?",
      "options": {
        "A": "High-density urban area",
        "B": "Low-density suburban area",
        "C": "Industrial area",
        "D": "Urban-rural transition zone"
      },
      "answer": "A",
      "answer_format": "mcq"
    },
    {
      "id": "q004",
      "type": "caption",
      "question": "Describe the urban scene characteristics of this area.",
      "answer": "This area is a typical high-density residential district with buildings arranged in a grid pattern, interspersed with green vegetation and connected by a road network.",
      "answer_format": "text"
    },
    {
      "id": "q005",
      "type": "caption",
      "question": "Provide a comprehensive description of the entire image and all visible objects.",
      "answer": "This aerial image shows a typical mixed-function urban district with high-density residential buildings, commercial facilities, and a well-developed road network. Green spaces and small water bodies are distributed locally, indicating a high level of urbanization.",
      "answer_format": "text"
    }
  ]
}
```

## Rules

- Only ask questions answerable from the provided region facts and image
- Do NOT mention region IDs or numbered regions in questions or answers
- Do NOT generate questions about change (this is segmentation, not change detection)
- Use descriptive location terms (north, south, center, etc.)
- If multiple regions of the same class exist, vary the questions
- Total questions must be at least 10
- Return raw JSON only
