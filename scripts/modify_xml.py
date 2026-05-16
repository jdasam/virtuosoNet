#!/usr/bin/env python3
import os
import glob
import subprocess
import re
from lxml import etree

# Information about Beethoven Piano Sonatas
# Format: Opus number: (Sonata name, Movement, Time Signature, QPM)
BEETHOVEN_SONATAS = {
    # Piano Sonata No. 3 in C major, Op. 2, No. 3
    "Op002No3": ("Piano Sonata No. 3 in C major", "Allegro con brio", (4, 4), 144),
    
    # Piano Sonata No. 5 in C minor, Op. 10, No. 1
    "Op010No1": ("Piano Sonata No. 5 in C minor", "Allegro molto e con brio", (3, 4), 210),
    
    # Piano Sonata No. 8 in C minor, Op. 13 "Pathétique"
    "Op013": ("Piano Sonata No. 8 in C minor 'Pathétique'", "Grave - Allegro di molto e con brio", (2, 2), 300),
    
    # Piano Sonata No. 9 in E major, Op. 14, No. 1
    "Op014No1": ("Piano Sonata No. 9 in E major", "Allegro", (4, 4), 120),
    
    # Piano Sonata No. 10 in G major, Op. 14, No. 2
    "Op014No2": ("Piano Sonata No. 10 in G major", "Allegro", (2, 4), 80),
    
    # Piano Sonata No. 12 in A-flat major, Op. 26 "Funeral March"
    "Op026": ("Piano Sonata No. 12 in A-flat major 'Funeral March'", "Andante con variazioni", (3, 8), 45),
    
    # Piano Sonata No. 15 in D major, Op. 28 "Pastoral"
    "Op028": ("Piano Sonata No. 15 in D major 'Pastoral'", "Allegro", (3, 4), 210),
    
    # Piano Sonata No. 16 in G major, Op. 31, No. 1
    "Op031No1": ("Piano Sonata No. 16 in G major", "Allegro vivace", (2, 4), 144),
    
    # Piano Sonata No. 18 in E-flat major, Op. 31, No. 3 "The Hunt"
    "Op031No3": ("Piano Sonata No. 18 in E-flat major 'The Hunt'", "Allegro", (3, 4), 130),
}

def extract_opus(filename):
    """Extract the opus number from a filename"""
    # Use regex to extract the opus number (e.g., Op002No3, Op031No1)
    match = re.search(r'Beethoven_(Op\d+(?:No\d+)?)', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def add_time_signature(xml_file):
    # Extract opus from filename to determine time signature
    opus = extract_opus(xml_file)
    # Default to 4/4 if opus not found in dictionary
    time_sig = BEETHOVEN_SONATAS.get(opus, (None, None, (4, 4), 120))[2]
    beats = time_sig[0]
    beat_type = time_sig[1]
    
    # Parse the XML file
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_file, parser)
    root = tree.getroot()
    
    # MusicXML namespace
    ns = root.nsmap.get(None, '')
    
    # Find all attributes elements
    modified = False
    for attributes in root.findall(f'.//{{{ns}}}attributes'):
        # Check if there's already a time signature in this attributes element
        time_elements = attributes.findall(f'{{{ns}}}time')
        
        # If no time signature exists, add one after the clef
        if not time_elements:
            # Find the last clef in the attributes
            clef_elements = attributes.findall(f'{{{ns}}}clef')
            
            if clef_elements:
                last_clef = clef_elements[-1]
                
                # Create time signature element
                time_elem = etree.Element(f'{{{ns}}}time')
                beats_elem = etree.SubElement(time_elem, f'{{{ns}}}beats')
                beats_elem.text = str(beats)
                beat_type_elem = etree.SubElement(time_elem, f'{{{ns}}}beat-type')
                beat_type_elem.text = str(beat_type)
                
                # Insert time after the last clef
                clef_idx = list(attributes).index(last_clef)
                attributes.insert(clef_idx + 1, time_elem)
                modified = True
    
    # Save the modified file if changes were made
    if modified:
        tree.write(xml_file, encoding='UTF-8', xml_declaration=True, pretty_print=True)
        print(f"Added time signature {beats}/{beat_type} to {xml_file}")
    else:
        print(f"No changes needed for {xml_file}")
    
    return xml_file

def run_inference(xml_file):
    # Extract opus number from filename
    opus = extract_opus(xml_file)
    
    # Get sonata information
    sonata_info = BEETHOVEN_SONATAS.get(opus, (None, None, (4, 4), 120))
    sonata_name = sonata_info[0]
    movement = sonata_info[1]
    qpm_primo = sonata_info[3]
    
    print(f"Running inference on {xml_file}")
    print(f"Sonata: {sonata_name}, Movement: {movement}, QPM: {qpm_primo}")
    
    # Format the command similar to run_inf.sh with qpm_primo
    command = f"""pipenv run python3 -m virtuoso --session_mode=inference \\
--checkpoint="../virtuosonet_checkpoints/yml_path=ymls/han_measnote_gru.yml meas_note=True delta_weight=5.0 delta_loss=True vel_balance_loss=True intermediate_loss=False_220203-122932/checkpoint_last.pt" \\
--model_code=beethoven \\
--xml_path={xml_file} \\
--composer=Beethoven \\
--output_path=icml_rebuttal/ \\
--qpm_primo={qpm_primo} \\
--
"""
    
    # Run the command
    print(f"Command: {command}")
    subprocess.run(command, shell=True)
    print(f"Completed inference on {xml_file}")

def main():
    # Get all MusicXML files in the directory
    xml_files = glob.glob("icml_rebuttal/*.musicxml")
    
    if not xml_files:
        print("No MusicXML files found in icml_rebuttal/ directory")
        return
    
    print(f"Found {len(xml_files)} MusicXML files")
    
    # Process each file
    for xml_file in xml_files:
        # Add time signature if needed
        modified_file = add_time_signature(xml_file)
        
        # Run inference on the modified file
        run_inference(modified_file)

if __name__ == "__main__":
    main()
