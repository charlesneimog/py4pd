from random import *
try:
    import pd
    pd_print = pd_print
except:
    pd_print = print


def sum(x, y):
    "It sums two numbers."
    x = int(x)
    y = int(y)
    return x + y

def arithm_ser(begin, end, step):
    "It calculates the arithmetic series."
    list = []
    for x in range(begin, end, step):
        list.append(x)
    return list

def fibonacci(n):
    "Calculate the nth fibonacci number."
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def thread_test():
    "It tests the threading module. Just return the hour after 5 seconds."
    import time
    import pd # import the py4pd module (embedded in the python interpreter)
    pd_print("Starting thread...")
    time.sleep(5)
    pd_print("Thread finished.")
    return time.strftime("%H:%M:%S")

def pd_output():
    "It sends some output to the py4pd output."
    import pd # import the py4pd module (embedded in the python interpreter)
    for x in range(10):
        pd.out(x)
    
def pd_message():
    "It sends a message to the py4pd message box."
    import pd # import the py4pd module (embedded in the python interpreter)
    pd_print("Hello from python!")
    return None

def pd_error():
    "It sends a message to the py4pd message box."
    import pd # import the py4pd module (embedded in the python interpreter)
    # NOT WORKING
    pd.error("Python error!")
    return None

def pd_send():
    "It sends a message to the py4pd message box."
    import pd
    import random
    pd.send()

def whereFiles():
    "It returns the path of the files."
    import os
    return os.__file__


def neoscoreTest():
    import os
    if os.name == 'posix':
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    elif os.name == 'nt':
        os.environ['QT_QPA_PLATFORM'] = 'windows:offscreen'

    elif os.name == 'mac':
        os.environ['QT_QPA_PLATFORM'] = 'cocoa:offscreen'


        
    script_dir = os.path.dirname(__file__)
        
    from neoscore.core.units import ZERO, Mm
    from neoscore.core import neoscore
    from neoscore.common import Staff, Clef, Barline, Chordrest, MusicText, Path, Font, Brush, Unit, Pen, barline_style
    from neoscore.core.text import Text
    from neoscore.western.chordrest import NoteheadTable
     
    neoscore.setup()
    pitch = 'c'
    alterations = ''
    octave = 4
    pitch_info = [(pitch, alterations, octave)]
    POSITION = (Mm(0), Mm(0))
    staff = Staff(POSITION, None, Mm(80))
    Clef(ZERO, staff, 'treble')
    # Articulações
    font = Font("Arial", Unit(9), italic=True)
    Path.rect((Mm(5), Mm(-14)), None, Mm(12), Mm(5),
              Brush.no_brush(), Pen(thickness=Mm(0.25)))  # rects
    Path.rect((Mm(18), Mm(-14)), None, Mm(12), Mm(5),
              Brush.no_brush(), Pen(thickness=Mm(0.25)))  # rects

    Text((Unit(20), staff.unit(-6)), staff, "ord.", font)
    MusicText((Unit(40), staff.unit(-6.5)), staff, "tremolo3", scale=0.8)

    MusicText((Unit(55), staff.unit(-6.3)), staff, "dynamicPP", scale=0.8)
    MusicText((Unit(70), staff.unit(-6.3)), staff,
              "dynamicFortePiano", scale=0.8)

    # Chave de repetição
    Barline(Mm(80), staff.group, barline_style.END)
    noteheads = NoteheadTable(
        "repeatDot",
        "repeatDot",
        "repeatDot",
        "repeatDot")
    note = [('a', '', 4)]
    Chordrest(Mm(76.5), staff, note, (int(1), int(1)), table=noteheads)
    note = [('c', '', 5)]
    Chordrest(Mm(76.5), staff, note, (int(1), int(1)), table=noteheads)
    Chordrest(Mm(5), staff, pitch_info, (int(1), int(1)))
    neoscore.render_image(
        rect=None,
        dest=f'{script_dir}/neoscoretest.png',
        wait=True,
        dpi=600)
    neoscore.shutdown()
    return "ok"

def runTest():
    import os
    import subprocess
    import sys
    if os.name == 'posix':
        cmd = 'pd -nogui -send "start-test bang"  test/test.pd'
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    elif os.name == 'nt':
        cmd = 'C:\Program Files\Pd\bin\pd.exe -send "start-test bang" -nogui py4pd_WIN64\test.pd'
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    elif os.name == 'mac':
        cmd = '/Applications/Pd-*.app/Contents/Resources/bin/pd -nogui -send "start-test bang" py4pd_macOS-Intel/test.pd'
        output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        outputLines = str(output).split('\\n')
        lastLine = outputLines[-2]
    # if lastLine contains "PASS" then the test passed
    if "PASS" in lastLine:
        # print in green
        print("\033[92m" + ' ALL TESTS PASSED ' + "\033[0m")
        return "ok"
    else:
        # split all the lines
        for line in outputLines:
            # if the line contains "FAIL" then print in red
            if "FAIL" in line:
                print("\033[91m" + line + "\033[0m")
            # if the line contains "PASS" then print in green
            elif "PASS" in line:
                print("\033[92m" + line + "\033[0m")
            # otherwise print normally
            else:
                print(line)
        sys.exit(1)
    
    
    
        
        
        
        
    
    