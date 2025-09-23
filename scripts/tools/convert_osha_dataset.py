#!/usr/bin/env python3
"""
Convert OSHA JSON dataset to PPE detection format
Extracts information from conversation-based JSON and creates proper annotation structure
"""

import json
import os
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import re

class OSHADatasetConverter:
    def __init__(self, json_file, images_dir, output_dir):
        """
        Initialize converter
        
        Args:
            json_file: Path to OSHA JSON file
            images_dir: Path to directory containing images
            output_dir: Output directory for converted dataset
        """
        self.json_file = json_file
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # PPE classes mapping
        self.ppe_classes = {
            'background': 0,
            'person': 1,
            'hard_hat': 2,
            'safety_vest': 3,
            'safety_gloves': 4,
            'safety_boots': 5,
            'eye_protection': 6,
            'no_hard_hat': 7,
            'no_safety_vest': 8,
            'no_safety_gloves': 9,
            'no_safety_boots': 10,
            'no_eye_protection': 11
        }
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        (self.output_dir / 'splits').mkdir(exist_ok=True)
        (self.output_dir / 'annotation_guides').mkdir(exist_ok=True)
        
    def load_json_data(self):
        """Load and parse the OSHA JSON file"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def parse_ppe_description(self, description):
        """
        Parse GPT description to extract PPE information
        
        Args:
            description: Text description of PPE status
            
        Returns:
            dict: Parsed PPE information
        """
        description = description.lower()
        
        # Extract PPE present and missing
        ppe_info = {
            'ppe_present': [],
            'ppe_missing': [],
            'violations': [],
            'scene_description': description
        }
        
        # Check for specific PPE items mentioned as present
        if 'hard hat' in description or 'helmet' in description:
            if 'wearing' in description and 'hard hat' in description:
                ppe_info['ppe_present'].append('hard_hat')
        
        if 'safety vest' in description or 'high-visibility' in description:
            if ('wearing' in description or 'vest' in description) and not 'missing' in description:
                ppe_info['ppe_present'].append('safety_vest')
        
        if 'safety goggles' in description or 'safety glasses' in description or 'eye protection' in description:
            if 'wearing' in description or 'present' in description:
                ppe_info['ppe_present'].append('eye_protection')
        
        if 'gloves' in description:
            if 'wearing' in description and 'gloves' in description:
                ppe_info['ppe_present'].append('safety_gloves')
        
        # Check for violations/missing PPE
        if 'no hard hat' in description or 'lacks a hard hat' in description or 'missing.*hard hat' in description:
            ppe_info['violations'].append('no_hard_hat')
            
        if 'no.*safety vest' in description or 'lacks.*vest' in description or 'missing.*vest' in description:
            ppe_info['violations'].append('no_safety_vest')
            
        # Always assume person is present in construction images
        ppe_info['ppe_present'].append('person')
        
        return ppe_info
    
    def get_image_dimensions(self, image_path):
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return (640, 480)  # Default size
    
    def create_annotation_template(self, image_id, image_filename, ppe_info, image_size):
        """
        Create XML annotation template with guidance
        
        Args:
            image_id: Image identifier
            image_filename: Image filename
            ppe_info: Parsed PPE information
            image_size: (width, height) tuple
        """
        width, height = image_size
        
        # Create XML structure
        annotation = ET.Element('annotation')
        
        # Add folder
        folder = ET.SubElement(annotation, 'folder')
        folder.text = 'osha_images'
        
        # Add filename
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_filename
        
        # Add size
        size = ET.SubElement(annotation, 'size')
        size_width = ET.SubElement(size, 'width')
        size_width.text = str(width)
        size_height = ET.SubElement(size, 'height')
        size_height.text = str(height)
        size_depth = ET.SubElement(size, 'depth')
        size_depth.text = '3'
        
        # Add guidance comment
        comment_text = f"\n<!-- ANNOTATION GUIDANCE FOR {image_filename} -->\n"
        comment_text += f"<!-- Scene: {ppe_info['scene_description'][:100]}... -->\n"
        comment_text += f"<!-- PPE Present: {', '.join(ppe_info['ppe_present'])} -->\n"
        comment_text += f"<!-- Violations: {', '.join(ppe_info['violations'])} -->\n"
        comment_text += "<!-- TODO: Add bounding boxes for each detected item -->\n"
        
        # Add placeholder objects for manual annotation
        all_items = ppe_info['ppe_present'] + ppe_info['violations']
        
        for item in set(all_items):  # Remove duplicates
            if item in self.ppe_classes:
                obj = ET.SubElement(annotation, 'object')
                
                name = ET.SubElement(obj, 'name')
                name.text = item
                
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                
                # Placeholder bounding box - NEEDS MANUAL ANNOTATION
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = '1'  # Placeholder
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = '1'  # Placeholder
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(width//4)  # Placeholder
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(height//4)  # Placeholder
        
        return annotation
    
    def save_annotation_guide(self, image_id, ppe_info):
        """Save human-readable annotation guide"""
        guide_content = f"ANNOTATION GUIDE FOR {image_id}\n"
        guide_content += "=" * 50 + "\n\n"
        guide_content += f"Scene Description:\n{ppe_info['scene_description']}\n\n"
        guide_content += f"PPE Items to Annotate:\n"
        
        all_items = set(ppe_info['ppe_present'] + ppe_info['violations'])
        for item in all_items:
            if item in self.ppe_classes:
                guide_content += f"- {item} (class {self.ppe_classes[item]})\n"
        
        guide_content += f"\nAnnotation Instructions:\n"
        guide_content += f"1. Open {image_id}.xml in LabelImg or similar tool\n"
        guide_content += f"2. Draw bounding boxes around each item listed above\n"
        guide_content += f"3. Make sure class names match exactly\n"
        guide_content += f"4. Pay special attention to violation classes (no_hard_hat, no_safety_vest)\n"
        
        guide_file = self.output_dir / 'annotation_guides' / f'{image_id}_guide.txt'
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
    
    def copy_images_to_output(self, data):
        """Copy images from osha_images to main images directory"""
        import shutil
        
        copied_images = []
        for item in data:
            image_filename = item['image']
            source_path = self.images_dir / image_filename
            dest_path = self.output_dir / 'images' / image_filename
            
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                copied_images.append(image_filename)
                print(f"Copied: {image_filename}")
            else:
                print(f"Missing image: {image_filename}")
        
        return copied_images
    
    def create_splits(self, image_list):
        """Create train/val/test splits"""
        random.shuffle(image_list)
        
        total = len(image_list)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        train_images = image_list[:train_end]
        val_images = image_list[train_end:val_end]
        test_images = image_list[val_end:]
        
        # Write split files
        for split_name, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            split_file = self.output_dir / 'splits' / f'{split_name}.txt'
            with open(split_file, 'w') as f:
                for img in images:
                    f.write(f"{img}\n")
            print(f"Created {split_name}.txt with {len(images)} images")
    
    def convert(self):
        """Main conversion process"""
        print("Loading OSHA JSON data...")
        data = self.load_json_data()
        
        print(f"Found {len(data)} entries in JSON file")
        
        # Copy images
        print("Copying images...")
        copied_images = self.copy_images_to_output(data)
        
        # Process each entry
        print("Creating annotation templates...")
        processed = 0
        
        for item in data:
            image_id = item['id']
            image_filename = item['image']
            
            # Skip if image wasn't copied
            if image_filename not in copied_images:
                continue
            
            # Find the GPT response
            gpt_response = None
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    gpt_response = conv['value']
                    break
            
            if not gpt_response:
                continue
            
            # Parse PPE information
            ppe_info = self.parse_ppe_description(gpt_response)
            
            # Get image dimensions
            image_path = self.output_dir / 'images' / image_filename
            image_size = self.get_image_dimensions(image_path)
            
            # Create annotation template
            annotation = self.create_annotation_template(
                image_id, image_filename, ppe_info, image_size
            )
            
            # Save XML file
            xml_file = self.output_dir / 'annotations' / f'{image_id}.xml'
            tree = ET.ElementTree(annotation)
            ET.indent(tree, space="  ", level=0)
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
            # Save annotation guide
            self.save_annotation_guide(image_id, ppe_info)
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} annotations...")
        
        # Create splits
        print("Creating train/val/test splits...")
        self.create_splits(copied_images)
        
        print(f"\nConversion complete!")
        print(f"- Processed {processed} images")
        print(f"- Created annotation templates in: {self.output_dir / 'annotations'}")
        print(f"- Created annotation guides in: {self.output_dir / 'annotation_guides'}")
        print(f"- Created split files in: {self.output_dir / 'splits'}")
        print(f"\nNext steps:")
        print(f"1. Use LabelImg to complete the bounding box annotations")
        print(f"2. Refer to annotation guides for each image")
        print(f"3. Replace placeholder coordinates with actual bounding boxes")


def main():
    # Paths
    json_file = "data/osha_florence_violation_check.json"
    images_dir = "data/images/osha_images"
    output_dir = "data"
    
    # Convert dataset
    converter = OSHADatasetConverter(json_file, images_dir, output_dir)
    converter.convert()

if __name__ == "__main__":
    main()