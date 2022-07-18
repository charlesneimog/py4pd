def cknnote(note, freq):
    import music21  
    from om_py.python_to_om import to_om
    import os
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
    cmd_line = go_to_folder + ' && "C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe" -s -fpng -dresolution=600 ' + str(ly_file)
    os.system(cmd_line)
    return ly_file
