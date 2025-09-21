
# PPE Dataset Structure

This directory should contain your PPE detection dataset in the following structure:

```
data/
├── images/              # All images (.jpg, .png)
├── annotations/         # Annotation files (.xml or .json)
└── splits/              # Train/val/test split files
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Annotation Formats Supported:

### 1. VOC XML Format (.xml)
```xml
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>hard_hat</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
```

### 2. JSON Format (.json)
```json
{
    "width": 640,
    "height": 480,
    "annotations": [
        {
            "class": "hard_hat",
            "bbox": [100, 100, 100, 100],
            "difficult": 0
        }
    ]
}
```

## PPE Classes:
- person: Worker/person in the image
- hard_hat: Hard hat/helmet
- safety_vest: High-visibility safety vest
- safety_gloves: Protective gloves
- safety_boots: Safety footwear
- eye_protection: Safety glasses/goggles
- no_hard_hat: Person without hard hat (violation)
- no_safety_vest: Person without safety vest (violation)
