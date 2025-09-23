#!/usr/bin/env python3
"""
Fix Label Studio tasks to avoid filename prefixes
"""

import json
import os
from pathlib import Path

def fix_label_studio_tasks():
    """Create a fixed version of label_studio_tasks.json"""
    
    # Load the current tasks file
    with open("label_studio_tasks.json", "r") as f:
        tasks = json.load(f)
    
    # Fix each task
    fixed_tasks = []
    for task in tasks:
        # Get the original image filename
        original_filename = task["data"]["image"].split("/")[-1]
        image_id = task["data"]["image_id"]
        
        # Create fixed task with direct image reference
        fixed_task = {
            "data": {
                "image": f"images/{original_filename}",  # Direct path without /data/local-files/
                "image_id": image_id,
                "annotation_guide": task["data"]["annotation_guide"]
            }
        }
        fixed_tasks.append(fixed_task)
    
    # Save fixed tasks
    with open("label_studio_tasks_fixed.json", "w") as f:
        json.dump(fixed_tasks, f, indent=2)
    
    print(f"✅ Created label_studio_tasks_fixed.json with {len(fixed_tasks)} tasks")
    print("🔄 Re-import this file in Label Studio to avoid filename prefixes")

def create_annotation_cleanup_script():
    """Create script to clean up exported annotations"""
    
    cleanup_script = '''#!/usr/bin/env python3
"""
Clean up Label Studio exported annotations
Removes filename prefixes and fixes paths
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re

def clean_exported_annotations(export_dir="exported_annotations", output_dir="data/annotations"):
    """
    Clean up exported XML files from Label Studio
    
    Args:
        export_dir: Directory containing exported XML files
        output_dir: Directory to save cleaned XML files
    """
    
    export_path = Path(export_dir)
    output_path = Path(output_dir)
    
    if not export_path.exists():
        print(f"❌ Export directory {export_dir} not found")
        print("1. Export annotations from Label Studio as Pascal VOC XML")
        print("2. Extract the zip file to 'exported_annotations' folder")
        print("3. Run this script again")
        return
    
    output_path.mkdir(exist_ok=True)
    
    cleaned = 0
    for xml_file in export_path.glob("*.xml"):
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get filename element
        filename_elem = root.find("filename")
        if filename_elem is not None:
            original_filename = filename_elem.text
            
            # Remove Label Studio prefix (format: prefix-filename)
            if "-" in original_filename:
                clean_filename = re.sub(r'^[a-f0-9]+-', '', original_filename)
                filename_elem.text = clean_filename
                
                # Save with clean filename
                output_file = output_path / f"{Path(clean_filename).stem}.xml"
                tree.write(output_file, encoding='utf-8', xml_declaration=True)
                
                print(f"✅ {original_filename} → {clean_filename}")
                cleaned += 1
    
    print(f"\\n🎉 Cleaned {cleaned} annotation files")
    print(f"📁 Saved to: {output_path}")
    print("✅ Ready for training!")

if __name__ == "__main__":
    clean_exported_annotations()
'''
    
    with open("scripts/clean_annotations.py", "w") as f:
        f.write(cleanup_script)
    
    print("✅ Created scripts/clean_annotations.py")

if __name__ == "__main__":
    fix_label_studio_tasks()
    create_annotation_cleanup_script()