powershell kill -n pd.com

cc -DPD -I "C:/Program Files/Pd/src" -I"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/include/" -L"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/libs/" -lpython310 -DMSW -DNT -DPD_LONGINTTYPE=__int64 -Wall -Wextra -Wshadow -Winline -Wstrict-aliasing -O3 -ffast-math -funroll-loops -fomit-frame-pointer -march=core2 -msse -msse2 -msse3 -mfpmath=sse   -o main.o -c py2pd.c

cc -static-libgcc -shared -Wl,--enable-auto-import "C:/Program Files/Pd/bin/pd.dll" "C:/Users/Neimog/AppData/Local/Programs/Python/Python310/python310.dll" -o py2pd.dll main.o