# Prompt: CD QA Generation from Caption

## Role Definition
You generate high-value question-answer pairs for remote sensing change detection, based on a **structured caption** produced by a previous analysis step.

## Input Facts
You will receive a JSON object containing:
- `caption`: a descriptive sentence following the template
  "The bi-temporal remote sensing images show that [changed object] is located [direction] of [reference object], with [surrounding object] distributed nearby. Compared with the earlier observation, [changed object] has undergone a noticeable change, while the surrounding area remains relatively stable."
- `changed_categories`: list of land-cover categories that changed
- `change_regions`: array of objects, each with `location`, `from_class`, `to_class`
- `num_change_regions`: total number of change regions
- `t1_class_counts`: exact count per source class (e.g., {"farmland": 3, "vegetation": 1})
- `t2_class_counts`: exact count per target class (e.g., {"building": 4})
- `change_type`: one of expansion / shrinkage / renewal / transformation
- `surrounding_categories`: nearby unchanged categories
- `reference_objects`: reference objects with locations

IMPORTANT:
- For counting questions, use `num_change_regions`, `t1_class_counts`, `t2_class_counts` as authoritative sources.
- Do NOT count regions yourself — use the pre-computed counts directly.
- Every location name and class name must come from the structured JSON only.

## Urban Change Classification System

| Change Type | Criteria |
|-------------|----------|
| **Expansion** | Farmland/vegetation/bare_land → building/road (new construction from non-urban land) |
| **Shrinkage** | Building → bare_land/vegetation/farmland (demolition, abandonment) |
| **Renewal** | Building → building (reconstruction) or bare_land → building in previously built area |
| **Transformation** | One land-use type changes to another different type (e.g., farmland → water) |

Sub-types for Expansion: edge expansion, leapfrog expansion, infill expansion
Sub-types for Shrinkage: demolition, abandonment

## Allowed Task Types

| Type | Meaning | Required answer style |
|------|---------|-----------------------|
| qa | binary confirmation | `yes` or `no` only |
| mcq | multiple choice | `A` / `B` / `C` / `D` only |
| caption | short description | 1-3 short sentences |

## Core Alignment Rules
- `qa` → `answer_format: "yes_no"`, answer exactly `yes` or `no`
- `mcq` → `answer_format: "mcq"`, exactly 4 options (A/B/C/D), answer one uppercase letter
- `caption` → `answer_format: "text"`, descriptive sentences covering urban change semantics
- Never use region IDs or numbered regions in output text
- Every location/class in questions must come from the structured JSON only

## Valuable Question Scope

### `qa` value scope (at least 3)
1. **CN** (Change Notification): `Did any visible change occur?`
2. **CtW** (Change to What): `Did any region become {class}?`
3. **CfW** (Change from What): `Did any changed region start as {class}?`
4. **Urban** (optional): `Did new construction appear in this image pair?`

### `mcq` value scope
1. `location`: `Where did {from_class} change to {to_class}?`
2. `change`: `What change happened in the {location}?`
3. `location_change`: `Which location-change pair is correct?`
4. `change_type` (at least 1): `What type of urban change is shown?`
   Options: urban expansion / urban shrinkage / urban renewal / land-use transformation / stable
5. `count` (optional): `How many change regions are there?` — MUST use counts as correct answer

### `caption` value scope — Three-Level Description

**Level 1: Global Description (at least 1)**
- Question: `Describe the overall urban change pattern in this area`
- Answer: urban context, direction of change, dominant pattern

**Level 2: Change Type (at least 1)**
- Question: `What type of urban change is occurring and what does it signify?`
- Answer: change type, cause, meaning

**Level 3: Detailed Description (at least 1)**
- Question: `Describe the specific change regions and their characteristics`
- Answer: number, location, content of each change region

**Local (at least 1)**: specific changed region with location and transition details

## Required Coverage (Core: 12 questions)

1. 1 QA `CN`
2. 1 QA `CtW`
3. 1 QA `CfW`
4. 1 MCQ `location`
5. 1 MCQ `change`
6. 1 MCQ `location_change`
7. 1 MCQ `change_type`
8. 1 Caption Level 1: global
9. 1 Caption Level 2: change type significance
10. 1 Caption Level 3: detailed
11. 1 Caption local
12. 1 Caption overall: comprehensive summary

After the core 12, you may append extra non-duplicate questions.

## Output Format

```json
{
  "qa": [
    {
      "id": "q001",
      "type": "qa",
      "answer_format": "yes_no",
      "question": "Did any visible change occur?",
      "answer": "yes"
    },
    {
      "id": "q002",
      "type": "mcq",
      "answer_format": "mcq",
      "question": "Where did farmland change to building?",
      "options": {"A": "southwest", "B": "north", "C": "east", "D": "south"},
      "answer": "A"
    },
    {
      "id": "q003",
      "type": "caption",
      "answer_format": "text",
      "question": "Describe the overall urban change pattern in this area",
      "answer": "This area exhibits significant urban expansion, with new buildings primarily in the southwest."
    }
  ]
}
```

## Output Constraints
- Output ONLY raw JSON
- No markdown code blocks
- Total questions must be at least 12
- Question text ≤ 20 words
- Caption answer ≤ 40 words
- Use exact lowercase class names: `building`, `road`, `vegetation`, `farmland`, `bareland`, `water`
- Prefer natural spatial expressions: north, south, east, west, northeast, northwest, southeast, southwest, center
