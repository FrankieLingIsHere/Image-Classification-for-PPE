
# Label Studio Setup Instructions for PPE Detection

## 1. Start Label Studio
Run the setup script:
```bash
# Windows
start_label_studio.bat

# Or manually:
label-studio start
```

## 2. Create Project
1. Open browser to http://localhost:8080
2. Sign up/Login (use any email)
3. Click "Create Project"
4. Name: "PPE Detection"
5. Description: "OSHA PPE Compliance Detection"

## 3. Configure Labels
In Project Settings > Labeling Interface, paste this configuration:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2" smart="true">
    <Label value="person" background="red"/>
    <Label value="hard_hat" background="blue"/>
    <Label value="safety_vest" background="yellow"/>
    <Label value="safety_gloves" background="green"/>
    <Label value="safety_boots" background="purple"/>
    <Label value="eye_protection" background="orange"/>
    <Label value="no_hard_hat" background="darkred"/>
    <Label value="no_safety_vest" background="darkgoldenrod"/>
  </RectangleLabels>
</View>
```

## 4. Import Data
1. Go to Project > Import
2. Upload `label_studio_tasks.json`
3. Click "Import"

## 5. Setup Storage
1. Go to Project Settings > Cloud Storage
2. Add Local Storage:
   - Storage Type: Local files
   - Absolute local path: `C:\Users\User\Documents\GitHub\ImageClassificationPPE\Image-Classification-for-PPE\data`
   - URL path prefix: `/data/local-files/`

## 6. Start Annotating!
1. Click on any task to start
2. Use the annotation guide shown in task details
3. Draw bounding boxes around PPE items
4. Use correct class labels
5. Save and move to next

## 7. Export Annotations
When done:
1. Go to Project > Export
2. Choose "Pascal VOC XML" format
3. Download and replace the placeholder annotations

## Annotation Tips:
- Use the annotation guide for each image
- Draw tight bounding boxes
- Pay attention to violation classes (no_hard_hat, no_safety_vest)
- Use zoom for precision
- Keyboard shortcuts: Space (next), Ctrl+Z (undo)

## Progress Tracking:
- Dashboard shows completion percentage
- Statistics show annotation quality
- Can filter completed/pending tasks
