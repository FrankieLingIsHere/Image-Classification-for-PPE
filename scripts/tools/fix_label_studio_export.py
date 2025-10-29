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


def rename_files_in_directory(directory, dry_run=False):
    """
    Simple utility to rename files in a directory by removing UUID prefixes.
    
    Args:
        directory: Directory containing files with UUID prefixes
        dry_run: If True, only show what would be renamed without actually renaming
    
    Examples:
        2e390e6e-image339.xml -> image339.xml
        81c078b4-image10.png -> image10.png
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    print(f"🔍 Scanning directory: {directory}")
    print()
    
    renamed_count = 0
    files_to_rename = []
    
    # Find all files with UUID prefix pattern (UUID-filename)
    for file_path in directory.iterdir():
        if file_path.is_file():
            filename = file_path.name
            
            # Check if filename has UUID prefix (e.g., "2e390e6e-image339.xml")
            if '-' in filename and len(filename.split('-')[0]) == 8:
                # Extract original name (everything after first dash)
                original_name = filename.split('-', 1)[1]
                new_path = file_path.parent / original_name
                
                files_to_rename.append((file_path, new_path, filename, original_name))
    
    if not files_to_rename:
        print("✅ No files with UUID prefixes found. All filenames look good!")
        return
    
    print(f"Found {len(files_to_rename)} files with UUID prefixes:")
    print()
    
    for old_path, new_path, old_name, new_name in files_to_rename:
        if dry_run:
            print(f"  {old_name} -> {new_name}")
        else:
            # Check if target already exists
            if new_path.exists():
                print(f"⚠️  Skipping {old_name} (target {new_name} already exists)")
                continue
            
            # Rename the file
            old_path.rename(new_path)
            renamed_count += 1
            print(f"✅ {old_name} -> {new_name}")
    
    print()
    if dry_run:
        print(f"💡 This was a dry run. Use --rename to actually rename {len(files_to_rename)} files.")
    else:
        print(f"✅ Successfully renamed {renamed_count} files!")
        if renamed_count < len(files_to_rename):
            print(f"⚠️  Skipped {len(files_to_rename) - renamed_count} files (targets already exist)")


def main():
    """Main function with usage instructions"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix Label Studio exports or rename files with UUID prefixes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix Label Studio export (zip/json)
  python fix_label_studio_export.py --export my_export.zip --output data/annotations
  
  # Preview renaming files in a directory (dry run)
  python fix_label_studio_export.py --rename-dir data/annotations --dry-run
  
  # Actually rename files in a directory
  python fix_label_studio_export.py --rename-dir data/annotations
  
  # Both modes remove UUID prefixes:
  # 2e390e6e-image339.xml -> image339.xml
        """
    )
    
    parser.add_argument('--export', type=str, help='Label Studio export file (.zip or .json)')
    parser.add_argument('--output', type=str, default='data/annotations', help='Output directory for fixed annotations')
    parser.add_argument('--rename-dir', type=str, help='Directory containing files to rename (removes UUID prefixes)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be renamed without actually renaming')
    
    args = parser.parse_args()
    
    print("🏷️  Label Studio Export Fixer")
    print("=" * 50)
    print()
    
    # Mode 1: Rename files in directory
    if args.rename_dir:
        rename_files_in_directory(args.rename_dir, dry_run=args.dry_run)
        return
    
    # Mode 2: Fix Label Studio export
    if args.export:
        fixer = LabelStudioExportFixer(args.export, args.output)
        fixer.fix_export()
        return
    
    # No arguments provided
    parser.print_help()

if __name__ == "__main__":
    main()