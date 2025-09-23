#!/usr/bin/env python3
"""
Fix Label Studio Export - Remove Prefixes and Restore Original Filenames
This script processes exported Label Studio annotations and fixes the filename prefixes
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import zipfile
import shutil
import re

class LabelStudioExportFixer:
    def __init__(self, export_file, output_dir="data/annotations"):
        """
        Initialize the export fixer
        
        Args:
            export_file: Path to Label Studio export file (.zip or .json)
            output_dir: Directory to save fixed annotations
        """
        self.export_file = Path(export_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_original_filename(self, prefixed_filename):
        """
        Extract original filename from Label Studio prefixed filename
        
        Examples:
        81c078b4-image10.png -> image10.png
        abc123def-image25.jpg -> image25.jpg
        """
        # Remove the prefix (everything before the first dash + dash)
        if '-' in prefixed_filename:
            return prefixed_filename.split('-', 1)[1]
        return prefixed_filename
    
    def fix_xml_annotation(self, xml_content, original_filename):
        """Fix XML annotation with correct filename"""
        try:
            root = ET.fromstring(xml_content)
            
            # Update filename in XML
            filename_elem = root.find('filename')
            if filename_elem is not None:
                filename_elem.text = original_filename
            
            # Update folder if needed
            folder_elem = root.find('folder')
            if folder_elem is not None:
                folder_elem.text = 'images'
            
            return ET.tostring(root, encoding='unicode')
        except Exception as e:
            print(f"Error fixing XML: {e}")
            return xml_content
    
    def process_voc_export(self, export_dir):
        """Process Pascal VOC XML export"""
        annotations_fixed = 0
        
        # Look for XML files in the export
        for xml_file in export_dir.rglob("*.xml"):
            try:
                # Read XML content
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract original filename from XML filename or path
                xml_filename = xml_file.name
                original_filename = self.extract_original_filename(xml_filename)
                
                # Fix the XML content
                fixed_content = self.fix_xml_annotation(content, original_filename)
                
                # Save with original filename
                output_file = self.output_dir / original_filename.replace('.png', '.xml').replace('.jpg', '.xml').replace('.jpeg', '.xml')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                print(f"✅ Fixed: {xml_filename} -> {output_file.name}")
                annotations_fixed += 1
                
            except Exception as e:
                print(f"❌ Error processing {xml_file}: {e}")
        
        return annotations_fixed
    
    def process_json_export(self, json_file):
        """Process JSON export and convert to XML"""
        annotations_fixed = 0
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                try:
                    # Extract image info
                    image_url = item.get('data', {}).get('image', '')
                    if not image_url:
                        continue
                    
                    # Get original filename
                    prefixed_filename = Path(image_url).name
                    original_filename = self.extract_original_filename(prefixed_filename)
                    
                    # Get image dimensions (you might need to read from actual image)
                    width = item.get('data', {}).get('width', 640)
                    height = item.get('data', {}).get('height', 480)
                    
                    # Create XML annotation
                    xml_content = self.create_xml_from_json(item, original_filename, width, height)
                    
                    # Save XML file
                    output_file = self.output_dir / original_filename.replace('.png', '.xml').replace('.jpg', '.xml').replace('.jpeg', '.xml')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(xml_content)
                    
                    print(f"✅ Converted: {prefixed_filename} -> {output_file.name}")
                    annotations_fixed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing item: {e}")
        
        except Exception as e:
            print(f"❌ Error reading JSON: {e}")
        
        return annotations_fixed
    
    def create_xml_from_json(self, item, filename, width, height):
        """Create XML annotation from JSON data"""
        
        # Create XML structure
        annotation = ET.Element('annotation')
        
        # Add folder
        folder = ET.SubElement(annotation, 'folder')
        folder.text = 'images'
        
        # Add filename
        filename_elem = ET.SubElement(annotation, 'filename')
        filename_elem.text = filename
        
        # Add size
        size = ET.SubElement(annotation, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, 'depth')
        depth_elem.text = '3'
        
        # Add objects from annotations
        annotations_list = item.get('annotations', [])
        for ann in annotations_list:
            for result in ann.get('result', []):
                if result.get('type') == 'rectanglelabels':
                    # Extract bounding box
                    value = result.get('value', {})
                    x = value.get('x', 0) * width / 100  # Convert from percentage
                    y = value.get('y', 0) * height / 100
                    w = value.get('width', 0) * width / 100
                    h = value.get('height', 0) * height / 100
                    
                    # Get label
                    labels = value.get('rectanglelabels', [])
                    if labels:
                        label = labels[0]
                        
                        # Create object element
                        obj = ET.SubElement(annotation, 'object')
                        
                        name = ET.SubElement(obj, 'name')
                        name.text = label
                        
                        difficult = ET.SubElement(obj, 'difficult')
                        difficult.text = '0'
                        
                        bndbox = ET.SubElement(obj, 'bndbox')
                        xmin = ET.SubElement(bndbox, 'xmin')
                        xmin.text = str(int(x))
                        ymin = ET.SubElement(bndbox, 'ymin')
                        ymin.text = str(int(y))
                        xmax = ET.SubElement(bndbox, 'xmax')
                        xmax.text = str(int(x + w))
                        ymax = ET.SubElement(bndbox, 'ymax')
                        ymax.text = str(int(y + h))
        
        # Convert to string with proper formatting
        ET.indent(annotation, space="  ", level=0)
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(annotation, encoding='unicode')
    
    def fix_export(self):
        """Main function to fix Label Studio export"""
        print("🔧 Fixing Label Studio Export...")
        print(f"📁 Export file: {self.export_file}")
        print(f"📂 Output directory: {self.output_dir}")
        print()
        
        if not self.export_file.exists():
            print(f"❌ Export file not found: {self.export_file}")
            return
        
        annotations_fixed = 0
        
        # Handle ZIP files
        if self.export_file.suffix.lower() == '.zip':
            print("📦 Processing ZIP export...")
            temp_dir = Path("temp_export")
            
            try:
                # Extract ZIP
                with zipfile.ZipFile(self.export_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Process XML files
                annotations_fixed = self.process_voc_export(temp_dir)
                
                # Clean up
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                print(f"❌ Error processing ZIP: {e}")
        
        # Handle JSON files
        elif self.export_file.suffix.lower() == '.json':
            print("📋 Processing JSON export...")
            annotations_fixed = self.process_json_export(self.export_file)
        
        else:
            print(f"❌ Unsupported file format: {self.export_file.suffix}")
            return
        
        print()
        print(f"🎉 Export fixed successfully!")
        print(f"📊 Processed {annotations_fixed} annotations")
        print(f"📁 Fixed annotations saved to: {self.output_dir}")
        print()
        print("✅ Your annotations are now ready for training!")
        print("💡 The original filenames have been restored (image1.xml, image2.xml, etc.)")


def main():
    """Main function with usage instructions"""
    import sys
    
    print("🏷️  Label Studio Export Fixer")
    print("=" * 50)
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fix_export.py <export_file>")
        print()
        print("Examples:")
        print("  python fix_export.py my_export.zip")
        print("  python fix_export.py my_export.json")
        print()
        print("The script will:")
        print("1. Remove Label Studio filename prefixes (81c078b4-image10.png -> image10.png)")
        print("2. Fix XML filenames to match your original images")
        print("3. Save corrected annotations to data/annotations/")
        return
    
    export_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/annotations"
    
    fixer = LabelStudioExportFixer(export_file, output_dir)
    fixer.fix_export()

if __name__ == "__main__":
    main()