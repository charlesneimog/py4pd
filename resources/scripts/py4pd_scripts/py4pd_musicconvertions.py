import pd
import math

def midicent2freq(midicent: float) -> float:
    """
    Converts a value in MIDI cents to its corresponding frequency in Hertz.

    Args:
    - midicent: A float representing the MIDI cent value to be converted to frequency.

    Returns:
    - A float representing the frequency in Hertz corresponding to the input MIDI cent value.
    """
    return 440 * (2 ** ((midicent - 6900) / 1200))


# =============================================================================
def freq2midicent(freq: float) -> float:
    """
    Converts a frequency in Hertz to its corresponding value in MIDI cents.

    Args:
    - freq: A float representing the frequency in Hertz to be converted to MIDI cents.

    Returns:
    - A float representing the MIDI cent value corresponding to the input frequency.
    """
    return 1200 * math.log2(freq / 440) + 6900


# =============================================================================
def midicent2note(midicent):

    """
    Converts a midicent note number to a note name string.
    """
    if isinstance(midicent, list):
        return [midicent2note(x) for x in midicent]
    sharpOrFlat = pd.getkey("accidentals")
    if sharpOrFlat is None or sharpOrFlat == "sharp":
        sharpOrFlat = "sharp"
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C']
    elif sharpOrFlat == "flat":
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C']
    else:
        raise ValueError("Invalid value for sharpOrFlat: " + str(sharpOrFlat))

    # multiply by 0.01, then round to nearest integer, then multiply by 100

    newmidicent = round(float(midicent) * 0.01) * 100
    desviation = midicent - newmidicent
    octave = int(midicent // 1200) - 1
    note = int(newmidicent / 100) % 12

    if desviation < 0:
        return f"{note_names[note]}{octave}-{abs(desviation)}¢"
    elif desviation > 0:
        return f"{note_names[note]}{octave}+{desviation}¢"
    else:
        return f"{note_names[note]}{octave}"










