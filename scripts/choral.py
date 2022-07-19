import os
import subprocess

def delete_by_extension(ext):
    folder = "C:/Users/Neimog/Git/py4pd"
    test = os.listdir(folder)
    for file in test:
        if file.endswith(ext):
            os.remove(folder + file)

def cknnote(note, freq):
    import music21  
    from om_py.python_to_om import to_om
    folder = "C:/Users/Neimog/Git/py4pd"
    us = music21.environment.UserSettings()
    us['lilypondPath'] = "C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe"
    folder_with_file_name = folder + str(freq) + '.ly'
    if freq < 293:
        clef = music21.clef.Treble8vbClef()
    else: 
        clef = music21.clef.TrebleClef()
    s = music21.stream.Stream([clef])
    s.insert(0, music21.note.Note(note, type='whole'))
    s.insert(0, music21.dynamics.Dynamic('pp'))
    ly_file = s.write('lilypond', fp=folder_with_file_name)
    go_to_folder = 'cd "' + folder + '"'
    cmd_line = go_to_folder + ' && "C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe" -s -fpng -dresolution=1200 ' + str(ly_file)
    subprocess.Popen(cmd_line, stdout=subprocess.PIPE, shell=True).stdout.read()
    return ly_file
