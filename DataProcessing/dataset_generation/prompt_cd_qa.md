# Prompt: Unified Q&A Pair Generation (Urban Change Understanding v2)

## Role Definition
You generate high-value question-answer pairs for remote sensing change detection, with a focus on **urban change understanding**. Beyond identifying *where* changes occurred, you classify *what type* of change happened and *what it means* for the urban landscape.

## Input Facts
The input contains:
- structured change facts derived from the image pair
- numbered bbox images for T1 / T2 / label

The structured fields include:
- `region_id`
- `location`
- `t1_class`
- `t2_class`
- `class_source`
- `num_regions`
- `t1_class_counts`: exact count per class in T1 (e.g., {"farmland": 3, "building": 1})
- `t2_class_counts`: exact count per class in T2 (e.g., {"building": 4})

Use the images to understand spatial layout and visible local changes.
Use the structured JSON fields as the only authority for `location`, `t1_class`, and `t2_class`.
For counting questions, use `t1_class_counts` / `t2_class_counts` as the authoritative source — do NOT count regions yourself.
Do not revise, correct, replace, or invent any location or class from the images.
The images are only for visual grounding and local appearance understanding.
Do not invent facts outside these inputs.

## Urban Change Classification System

Before generating questions, classify the overall change pattern into one of these types:

| Change Type | Criteria |
|-------------|----------|
| **Expansion** | Farmland/vegetation/bare_land → building/highway (new construction from non-urban land) |
| **Shrinkage** | Building → bare_land/vegetation/farmland (demolition, abandonment) |
| **Renewal** | Building → building (reconstruction within built area) or bare_land → building in previously built area |
| **Transformation** | One land-use type changes to another different type (e.g., farmland → water, vegetation → highway) |
| **Stable** | No significant change (rare in change detection data) |

### Sub-types for Expansion
- **Edge expansion**: new construction at the urban boundary
- **Leapfrog expansion**: new construction isolated from existing built area
- **Infill expansion**: new construction filling gaps within built area

### Sub-types for Shrinkage
- **Demolition**: buildings removed
- **Abandonment**: industrial or built areas becoming vacant

## Allowed Task Types
Use only these 3 task categories in the output `type` field:

| Type | Meaning | Required answer style |
|------|---------|-----------------------|
| qa | binary confirmation | `yes` or `no` only |
| mcq | multiple choice | `A` / `B` / `C` / `D` only |
| caption | short description | 1-3 short sentences |

Do not output any other `type` values.

## Core Alignment Rules
- `qa` must always use `answer_format: "yes_no"`
- `qa` answers must be exactly `yes` or `no`
- `mcq` must always use `answer_format: "mcq"`
- `mcq` must always contain exactly 4 options with keys `A`, `B`, `C`, `D`
- `mcq` answers must be exactly one uppercase letter: `A`, `B`, `C`, or `D`
- `caption` must always use `answer_format: "text"`
- `caption` answers must be descriptive sentences covering urban change semantics
- `region_id` is only for internal alignment with the numbered bbox images and must never appear in output text
- Every location name and class name in questions, options, and answers must be copied from the structured JSON only
- Wrong MCQ options may recombine provided locations/classes, but may not introduce unseen locations or unseen class names

## Valuable Question Scope
Generate questions grounded in visible change facts.

### `qa` value scope
Use `qa` for these semantic slots (at least 3 questions):

1. **CN** (Change Notification)
   - `Did any visible change occur?`

2. **CtW** (Change to What)
   - `Did any region become {class}?`

3. **CfW** (Change from What)
   - `Did any changed region start as {class}?`

4. **Urban Expansion** (optional, when applicable)
   - `Did new construction appear in this image pair?`

### `mcq` value scope
Use `mcq` for these subtypes:

1. `location`: `Where did {source_class} change to {target_class}?`
2. `change`: `What change happened in the {location}?`
3. `location_change`: `Which location-change pair is correct?`
4. `change_type` (at least 1): `What type of urban change is shown in this image pair?`
   - Options should include: urban expansion / urban shrinkage / urban renewal / land-use transformation / stable
5. `count` (optional): `How many change regions are there?` or `How many regions changed from {class}?`
   - MUST use `num_regions` / `t1_class_counts` / `t2_class_counts` as the correct answer

### `caption` value scope — Three-Level Urban Change Description
Use `caption` for urban change understanding at three levels:

**Level 1: Global Urban Description (at least 1)**
- Question: `Describe the overall urban change pattern in this area`
- Answer should cover: urban context, direction of change, dominant pattern
- Example: `This area shows significant urban expansion, with new buildings primarily appearing in the eastern and southern regions, converted from former farmland and vegetation.`

**Level 2: Change Type Classification (at least 1)**
- Question: `What type of urban change is occurring and what does it signify?`
- Answer should cover: change type (expansion/shrinkage/renewal/transformation), cause, and meaning
- Example: `The dominant change is urban expansion through edge development, where farmland on the urban periphery is being converted to new building construction.`
- OR: `This area shows urban renewal, with older structures in the center being demolished and replaced with new buildings.`

**Level 3: Mask-based Detailed Description (at least 1)**
- Question: `Describe the specific change regions and their characteristics`
- Answer should cover: number of change regions, location and content of each
- Example: `Three main change regions detected: new building complex in the north, new road construction in the southeast, and partial building demolition in the west.`
- OR: `Changed areas cover approximately 15% of the image, with new construction area significantly exceeding demolition area.`

**Local Description (at least 1)**
- Focus on a specific changed region with location and transition details

## Generation Constraints
- Output all items inside a single top-level `qa` array
- Use a two-stage generation flow
- Stage 1: generate the core questions (at least 12)
- Stage 2: append extra high-value non-duplicate questions
- The final total must be at least 12 questions
- Use exact lowercase class names only:
  - `building`, `highway`, `vegetation`, `farmland`, `bare_land`, `water`
- Never use region ids, numbered regions, `region_1`, `region 2`, or `area 3`
- Prefer natural spatial expressions: north, south, east, west, northeast, northwest, southeast, southwest, center
- Question text length: at most 20 words
- `caption` answer length: at most 40 words, 1-2 sentences for Level 1-2, 2-3 sentences for Level 3

## Required Coverage (Core Block: 12 questions)

1. 1 QA `CN` question
2. 1 QA `CtW` question
3. 1 QA `CfW` question
4. 1 MCQ `location` question
5. 1 MCQ `change` question
6. 1 MCQ `location_change` question
7. 1 MCQ `change_type` question (urban change classification)
8. 1 Caption Level 1: global urban description
9. 1 Caption Level 2: change type and significance
10. 1 Caption Level 3: mask-based detailed description
11. 1 Caption local: specific region change description
12. 1 Caption overall: comprehensive change summary

After the core block, you may append extra questions that:
- are answerable from the provided facts
- are non-duplicate
- use only `qa`, `mcq`, or `caption`
- keep the same answer-format rules

## Output Example
```json
{
  "qa": [
    {
      "id": "sample_001_q001",
      "type": "qa",
      "answer_format": "yes_no",
      "question": "Did any visible change occur?",
      "answer": "yes"
    },
    {
      "id": "sample_001_q002",
      "type": "qa",
      "answer_format": "yes_no",
      "question": "Did any region become building?",
      "answer": "yes"
    },
    {
      "id": "sample_001_q003",
      "type": "qa",
      "answer_format": "yes_no",
      "question": "Did any changed region start as farmland?",
      "answer": "yes"
    },
    {
      "id": "sample_001_q004",
      "type": "mcq",
      "answer_format": "mcq",
      "question": "Where did farmland change to building?",
      "options": {
        "A": "southwest",
        "B": "north",
        "C": "east",
        "D": "south"
      },
      "answer": "A"
    },
    {
      "id": "sample_001_q005",
      "type": "mcq",
      "answer_format": "mcq",
      "question": "What change happened in the south?",
      "options": {
        "A": "vegetation to building",
        "B": "farmland to water",
        "C": "building to bare_land",
        "D": "water to highway"
      },
      "answer": "A"
    },
    {
      "id": "sample_001_q006",
      "type": "mcq",
      "answer_format": "mcq",
      "question": "Which location-change pair is correct?",
      "options": {
        "A": "southwest: farmland to building",
        "B": "south: water to farmland",
        "C": "west: building to water",
        "D": "east: bare_land to highway"
      },
      "answer": "A"
    },
    {
      "id": "sample_001_q007",
      "type": "mcq",
      "answer_format": "mcq",
      "question": "What type of urban change is shown in this image pair?",
      "options": {
        "A": "Urban expansion",
        "B": "Urban shrinkage",
        "C": "Urban renewal",
        "D": "Land-use transformation"
      },
      "answer": "A"
    },
    {
      "id": "sample_001_q008",
      "type": "caption",
      "answer_format": "text",
      "question": "Describe the overall urban change pattern in this area",
      "answer": "This area exhibits significant urban expansion, with new buildings primarily concentrated in the southwest and south, converted from former farmland and vegetation."
    },
    {
      "id": "sample_001_q009",
      "type": "caption",
      "answer_format": "text",
      "question": "What type of urban change is occurring and what does it signify?",
      "answer": "The dominant pattern is edge expansion, where agricultural land on the urban periphery is being converted to new building construction, indicating active urbanization."
    },
    {
      "id": "sample_001_q010",
      "type": "caption",
      "answer_format": "text",
      "question": "Describe the specific change regions and their characteristics",
      "answer": "Two main change regions detected: a large new building complex in the southwest converted from farmland, and a smaller vegetation-to-building transition in the south."
    },
    {
      "id": "sample_001_q011",
      "type": "caption",
      "answer_format": "text",
      "question": "Summarize the change in the southwest",
      "answer": "In the southwest, farmland changed to building."
    },
    {
      "id": "sample_001_q012",
      "type": "caption",
      "answer_format": "text",
      "question": "Describe the overall change pattern",
      "answer": "Visible changes appeared in the southwest and the south, both involving conversion to new building construction."
    }
  ]
}
```

## Output Constraints
- Output ONLY raw JSON
- No markdown code blocks in the final answer
- No explanations
- The first 12 items in the `qa` array must be the core block
- Any later items are optional expansion questions
