import os
import mido
from pathlib import Path

def elongate_notes_with_pedal(input_midi_path, output_midi_path=None):
    """
    Elongate notes in a MIDI file based on sustain pedal (CC 64) activity.
    If pedal is on, note-off events are delayed until the pedal is off.
    
    Args:
        input_midi_path: Path to the input MIDI file
        output_midi_path: Path to save the modified MIDI file (default: input_path with _elongated suffix)
    
    Returns:
        Path to the output MIDI file
    """
    if output_midi_path is None:
        # Create output filename by adding _elongated before the extension
        file_path = Path(input_midi_path)
        output_midi_path = str(file_path.with_stem(f"{file_path.stem}_elongated"))
    
    # Load the MIDI file
    midi_file = mido.MidiFile(input_midi_path)
    
    # Create a new MIDI file with the same properties
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
    
    # Process each track
    for track_idx, track in enumerate(midi_file.tracks):
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)
        
        # Keep track of active notes and when the pedal is on
        active_notes = {}  # {(channel, note): original note_off_message}
        pedal_on = False
        delayed_note_offs = []  # Store note-offs that should be delayed due to pedal
        
        # Process messages in the track
        for msg in track:
            # Pass through all non-note and non-pedal messages unchanged
            if not msg.type.startswith('note_') and not (msg.type == 'control_change' and msg.control == 64):
                new_track.append(msg)
                continue
            
            # Handle note_on messages
            if msg.type == 'note_on' and msg.velocity > 0:
                # Add the note to active notes
                new_track.append(msg)
                active_notes[(msg.channel, msg.note)] = None
            
            # Handle note_off messages (also note_on with velocity 0)
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                note_key = (msg.channel, msg.note)
                
                # If pedal is on, delay the note_off event
                if pedal_on and note_key in active_notes:
                    delayed_note_offs.append(msg)
                    active_notes[note_key] = msg  # Store the original note_off message
                else:
                    # Pedal is off, send note_off immediately
                    new_track.append(msg)
                    if note_key in active_notes:
                        del active_notes[note_key]
            
            # Handle sustain pedal messages
            elif msg.type == 'control_change' and msg.control == 64:
                new_track.append(msg)
                
                # Check if pedal state is changing
                was_pedal_on = pedal_on
                pedal_on = msg.value >= 64
                
                # If pedal is turning off, release all delayed notes
                if was_pedal_on and not pedal_on:
                    for note_off_msg in delayed_note_offs:
                        new_track.append(note_off_msg)
                        note_key = (note_off_msg.channel, note_off_msg.note)
                        if note_key in active_notes:
                            del active_notes[note_key]
                    delayed_note_offs = []
        
        # Add any remaining note_offs at the end (shouldn't normally happen)
        for note_off_msg in delayed_note_offs:
            new_track.append(note_off_msg)
    
    # Save the modified MIDI file
    new_midi.save(output_midi_path)
    return output_midi_path

def process_all_midi_files(directory_path):
    """Process all MIDI files in the specified directory."""
    processed_files = []
    
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist!")
        return processed_files
    
    # Process each MIDI file
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.mid', '.midi')):
            input_path = os.path.join(directory_path, file)
            output_path = elongate_notes_with_pedal(input_path)
            processed_files.append((input_path, output_path))
            print(f"Processed: {input_path} -> {output_path}")
    
    return processed_files

if __name__ == "__main__":
    # Directory containing MIDI files to process
    midi_directory = "icml_rebuttal"
    
    # Process all MIDI files in the directory
    processed_files = process_all_midi_files(midi_directory)
    
    if processed_files:
        print(f"\nSuccessfully processed {len(processed_files)} MIDI files.")
    else:
        print("\nNo MIDI files were processed.")
