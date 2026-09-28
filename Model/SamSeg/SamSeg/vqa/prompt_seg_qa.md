# SEG QA Generation Prompt — Urban Scene Understanding v3

You are generating a visual question-answering dataset for remote sensing segmentation images, based on a **structured caption** produced by a previous analysis step.

## Input Facts
You will receive a JSON object containing:
- `caption`: a descriptive sentence following the template
  "The remote sensing image shows that [target object] is located [direction] of [reference object], with [surrounding object] distributed nearby. Together, these land-cover elements form the spatial structure of the scene."
- `present_categories`: list of land-cover categories present in the image
- `category_details`: array of objects, each with `category`, `location`, `description`
- `num_regions`: total number of valid regions
- `class_counts`: exact count per class (e.g., {"building": 5, "vegetation": 3})
- `scene_type`: one of the unified scene types below
- `surrounding_categories`: nearby categories

IMPORTANT: For any counting question, you MUST use `class_counts` as the authoritative source.
Do NOT count regions yourself — use the pre-computed values directly.

## Urban Scene Understanding Dimensions

When analyzing the scene, consider these 7 dimensions:

### 1. Built Environment (建成环境)
- Building density: high/medium/low
- Building scale: large/medium/small
- Building layout: grid/organic/scattered/clustered

### 2. Transportation Network (交通网络)
- Main roads, highways, intersections
- Road network density and connectivity

### 3. Green Space & Ecology (绿地生态)
- Parks, forests, urban greenery
- Vegetation coverage and distribution

### 4. Water Environment (水体环境)
- Rivers, lakes, reservoirs, canals
- Water body impact on spatial layout

### 5. Industry & Commerce (工业与商业)
- Industrial parks, warehouses, commercial centers
- Large-scale structures with regular layout

### 6. Residential Distribution (居住区分布)
- High-rise residential, villa areas, urban villages
- Settlement patterns and density

### 7. Urban-Rural Pattern (城乡格局)
- Urban core, urban fringe, suburbs, rural settlements
- Transition zone characteristics

## Unified Urban Scene Classification

Use these scene type labels:
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

Generate a JSON object with a `qa` array containing at least 10 questions:

### 1. QA (Yes/No) — at least 3
- **CN**: "Is there any visible object in this image?" → "yes" (always)
- **CtW**: "Is there a {class} in this image?" → based on present_categories
- **CfW**: "Is the object in the {location} a {class}?" → based on category_details
- **Scene**: "Does this image show a high-density urban area?" → based on scene_type

Rules: answer_format: "yes_no", answer exactly "yes" or "no"

### 2. MCQ (Multiple Choice) — at least 4
- **location**: "Where is the {class} located?" → 4 directions, one correct
- **class**: "What type of object is located in the {location}?" → 4 classes, one correct
- **count**: "How many {class} regions are visible?" → MUST use `class_counts`
- **scene_type** (at least 1): "What type of urban scene is shown?" → 4 scene types

Rules: answer_format: "mcq", exactly 4 options (A/B/C/D), answer one letter

### 3. Caption (Descriptive) — at least 3
- **local**: "Describe the {class} in the {location} of the image."
- **scene_understanding** (at least 1): "Describe the urban scene characteristics."
  Answer: building density, layout pattern, dominant land use
- **overall**: "Provide a comprehensive description of the entire image."

Rules: answer_format: "text", non-empty descriptive text

## Output Format

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
      "answer": "This area is a typical high-density residential district with buildings arranged in a grid pattern.",
      "answer_format": "text"
    }
  ]
}
```

## Rules
- Only ask questions answerable from the provided caption and structured facts
- Do NOT generate change-related questions (this is segmentation, not change detection)
- Use descriptive location terms (north, south, center, etc.)
- If multiple regions of the same class exist, vary the questions
- Total questions ≥ 10
- Output raw JSON only
- Use exact lowercase class names: `building`, `road`, `vegetation`, `farmland`, `bareland`, `water`
