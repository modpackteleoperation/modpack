#!/usr/bin/env python3
"""
Clean up URDF joint definitions by removing numerical artifacts
and correcting coordinate frame issues.
"""

import re
import xml.etree.ElementTree as ET
import numpy as np

def clean_numerical_artifacts(value_str, threshold=1e-10):
    """Round tiny floating point values to zero."""
    try:
        value = float(value_str)
        if abs(value) < threshold:
            return "0"
        # Round to 6 decimal places to remove artifacts
        return f"{value:.6f}".rstrip('0').rstrip('.')
    except ValueError:
        return value_str

def clean_joint_origin(origin_element):
    """Clean up joint origin xyz and rpy attributes."""
    
    # Clean xyz coordinates
    if 'xyz' in origin_element.attrib:
        xyz_str = origin_element.attrib['xyz']
        xyz_values = xyz_str.split()
        cleaned_xyz = [clean_numerical_artifacts(val) for val in xyz_values]
        origin_element.attrib['xyz'] = ' '.join(cleaned_xyz)
    
    # Clean rpy rotations
    if 'rpy' in origin_element.attrib:
        rpy_str = origin_element.attrib['rpy']
        rpy_values = rpy_str.split()
        cleaned_rpy = []
        
        for val in rpy_values:
            cleaned_val = clean_numerical_artifacts(val)
            # Convert to float for further processing
            angle = float(cleaned_val)
            
            # Normalize angles to [-pi, pi] range
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            
            # Round common angles to exact values
            if abs(angle) < 1e-6:
                cleaned_rpy.append("0")
            elif abs(angle - np.pi/2) < 1e-6:
                cleaned_rpy.append("1.5708")
            elif abs(angle + np.pi/2) < 1e-6:
                cleaned_rpy.append("-1.5708")
            elif abs(angle - np.pi) < 1e-6:
                cleaned_rpy.append("3.14159")
            elif abs(angle + np.pi) < 1e-6:
                cleaned_rpy.append("-3.14159")
            else:
                cleaned_rpy.append(f"{angle:.6f}".rstrip('0').rstrip('.'))
        
        origin_element.attrib['rpy'] = ' '.join(cleaned_rpy)

def apply_joint_corrections(root):
    """Apply specific corrections for known problematic joints."""
    
    joint_corrections = {
        'r1': {
            # This joint has a ~135° rotation that should probably be 0
            'xyz': '0.04925 -0.0549966 -0.0349605',  # Keep position
            'rpy': '0 0 0'  # Zero out the problematic rotation
        },
        # Add other joint corrections as needed
    }
    
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        if joint_name in joint_corrections:
            origin = joint.find('origin')
            if origin is not None:
                corrections = joint_corrections[joint_name]
                if 'xyz' in corrections:
                    origin.set('xyz', corrections['xyz'])
                if 'rpy' in corrections:
                    origin.set('rpy', corrections['rpy'])
                print(f"Applied corrections to joint '{joint_name}'")

def clean_urdf_file(input_file, output_file, apply_corrections=True):
    """Clean up a URDF file and save the result."""
    
    # Parse the URDF
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Clean all joint origins
    joint_count = 0
    for joint in root.findall('.//joint'):
        origin = joint.find('origin')
        if origin is not None:
            print(f"Cleaning joint: {joint.get('name')}")
            clean_joint_origin(origin)
            joint_count += 1
    
    # Apply specific corrections for problematic joints
    if apply_corrections:
        apply_joint_corrections(root)
    
    # Write the cleaned URDF
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Cleaned {joint_count} joints and saved to {output_file}")

def analyze_joint_rotations(urdf_file):
    """Analyze joint rotations to identify potential issues."""
    
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    print("Joint Rotation Analysis:")
    print("=" * 50)
    
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        origin = joint.find('origin')
        if origin is not None and 'rpy' in origin.attrib:
            rpy_str = origin.attrib['rpy']
            rpy_values = [float(val) for val in rpy_str.split()]
            
            # Convert to degrees for easier interpretation
            rpy_degrees = [np.degrees(angle) for angle in rpy_values]
            
            print(f"Joint '{joint_name}' ({joint_type}):")
            print(f"  RPY (rad): {rpy_values}")
            print(f"  RPY (deg): {rpy_degrees}")
            
            # Flag potentially problematic rotations
            for i, angle_deg in enumerate(rpy_degrees):
                if abs(angle_deg) > 5 and abs(angle_deg) not in [90, 180, 270, 360]:
                    axis_name = ['roll', 'pitch', 'yaw'][i]
                    print(f"  ⚠️  Large {axis_name} rotation: {angle_deg:.1f}°")
            print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clean_urdf_joints.py <input_urdf> [output_urdf]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.urdf', '_cleaned.urdf')
    
    print(f"Analyzing {input_file}...")
    analyze_joint_rotations(input_file)
    
    print(f"Cleaning {input_file}...")
    clean_urdf_file(input_file, output_file)
    
    print(f"\nRe-analyzing {output_file}...")
    analyze_joint_rotations(output_file)